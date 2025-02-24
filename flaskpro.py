from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO
import eventlet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, time, datetime, os, requests, scipy.io
from scipy.io import loadmat
import plotly.graph_objects as go

app = Flask(__name__, template_folder="templates")
app.secret_key = 'secretkey'
socketio = SocketIO(app, async_mode='eventlet')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------------------------------------------------
# Global API Keys (replace with your actual keys)
# -----------------------------------------------------------------------------
OPENWEATHER_API_KEY = "8e3ac63e6a6f68540b02fde7be6e8852"
WEATHERSTACK_API_KEY = "8899eaaa3bbc0952d0d9191eaf00b62a"

# -----------------------------------------------------------------------------
# Simulation Classes (traits from your newtry.py)
# -----------------------------------------------------------------------------
def fetch_historical_weather_tomorrow(city, date):
    """
    Fetches historical weather data (cloud cover, wind speed, and precipitation intensity)
    for the given city and date using the Tomorrow.io API.

    Parameters:
        city (str): City name (will be geocoded to lat/lon using get_lat_lon).
        date (str): Date in YYYY-MM-DD format.

    Returns:
        tuple: (irradiances, wind_speeds, water_flows) each as a list of hourly values,
               or None if the fetch fails.
    """
    # Get latitude and longitude for the city (using your existing get_lat_lon function)
    lat, lon = get_lat_lon(city)
    if lat is None or lon is None:
        return None

    # Prepare startTime and endTime in ISO format (UTC assumed)
    startTime = f"{date}T00:00:00Z"
    endTime = f"{date}T23:59:59Z"
    # The location is provided as "lat,lon"
    location_str = f"{lat},{lon}"

    url = "https://api.tomorrow.io/v4/timelines"
    params = {
        "location": location_str,
        "fields": "cloudCover,windSpeed,precipitationIntensity",
        "timesteps": "1h",
        "units": "metric",
        "startTime": startTime,
        "endTime": endTime,
        "apikey": "jWafTw6FoD8dMmSXCG5Odg16Mdl3rRL9"  # Tomorrow.io API key provided by you
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Navigate the JSON structure to get the hourly intervals
        intervals = data.get("data", {}).get("timelines", [{}])[0].get("intervals", [])
        if not intervals:
            return None

        irradiances = []
        wind_speeds = []
        water_flows = []
        for interval in intervals:
            values = interval.get("values", {})
            # Use cloudCover to compute an estimated irradiance (max 1000 W/m^2 when clear)
            cloudCover = values.get("cloudCover", 50)
            irradiance = max(0, 1000 * (1 - cloudCover / 100))
            irradiances.append(irradiance)
            # Get windSpeed (in m/s)
            wind_speed = values.get("windSpeed", 5)
            wind_speeds.append(wind_speed)
            # Use precipitationIntensity (in mm/hr) to calculate a water flow estimate (multiplying by 100, for example)
            precip = values.get("precipitationIntensity", 0)
            water_flow = precip * 100
            water_flows.append(water_flow)
        return irradiances, wind_speeds, water_flows
    else:
        return None

class House:
    def __init__(self, name, base_load):
        self.name = name
        self.base_load = base_load

    def get_load(self, hour):
        variation = 1 + 0.5 * np.sin(np.pi * (hour - 6) / 12)
        load = self.base_load * variation + np.random.randint(-50, 50)
        return max(load, 0)

class WeatherPredictor:
    def __init__(self, max_irradiance=1000, avg_wind=8, avg_water_flow=500):
        self.max_irradiance = max_irradiance
        self.avg_wind = avg_wind
        self.avg_water_flow = avg_water_flow

    def get_weather(self, hour):
        if 6 <= hour <= 18:
            irradiance = self.max_irradiance * np.sin(np.pi * (hour - 6) / 12)
        else:
            irradiance = 0
        wind_speed = max(np.random.normal(self.avg_wind, 2), 0)
        water_flow = self.avg_water_flow + np.random.normal(0, 20)
        return {"irradiance": irradiance, "wind_speed": wind_speed, "water_flow": water_flow}

class SolarPanel:
    def __init__(self, area=10, efficiency=0.15):
        self.area = area
        self.efficiency = efficiency

    def generate_power(self, weather):
        base_power = self.efficiency * self.area * weather["irradiance"]
        return base_power * np.random.uniform(0.8, 1.2)

class WindTurbine:
    def __init__(self, rotor_diameter=5, efficiency=0.4, air_density=1.225):
        self.rotor_diameter = rotor_diameter
        self.efficiency = efficiency
        self.air_density = air_density
        self.swept_area = np.pi * (rotor_diameter / 2) ** 2

    def generate_power(self, weather):
        wind_speed = weather["wind_speed"]
        if wind_speed < 3 or wind_speed > 25:
            return 0
        return 0.5 * self.air_density * self.swept_area * (wind_speed ** 3) * self.efficiency

class HydroGenerator:
    def __init__(self, efficiency=0.7, head=10, gravity=9.81):
        self.efficiency = efficiency
        self.head = head
        self.gravity = gravity

    def generate_power(self, weather):
        water_flow = weather["water_flow"] / 1000.0
        return water_flow * self.gravity * self.head * self.efficiency * 1000

class BatteryStorage:
    def __init__(self, capacity=10000, charge_rate=2000, discharge_rate=2000):
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.current_energy = capacity / 2

    def charge(self, energy, dt=1):
        max_charge = self.charge_rate * dt
        energy = min(energy, max_charge)
        charge_possible = self.capacity - self.current_energy
        actual_charge = min(energy, charge_possible)
        self.current_energy += actual_charge
        return actual_charge

    def discharge(self, energy, dt=1):
        max_discharge = self.discharge_rate * dt
        energy = min(energy, max_discharge)
        actual_discharge = min(energy, self.current_energy)
        self.current_energy -= actual_discharge
        return actual_discharge

class BackupGenerator:
    def __init__(self, max_output=5000):
        self.max_output = max_output
        self.total_output = 0
        self.active = False

    def generate_power(self, deficit):
        output = min(deficit, self.max_output)
        self.total_output += output
        return output

    def reset(self):
        self.active = False

class EnergyManagementSystem:
    def __init__(self, mat_params):
        self.weather_predictor = WeatherPredictor(
            max_irradiance=mat_params.get('weather_max_irradiance', 1000),
            avg_wind=mat_params.get('weather_avg_wind', 8),
            avg_water_flow=mat_params.get('weather_avg_water_flow', 500)
        )
        self.solar = SolarPanel(
            area=mat_params.get('solar_area', 10),
            efficiency=mat_params.get('solar_efficiency', 0.15)
        )
        self.wind = WindTurbine(
            rotor_diameter=mat_params.get('wind_rotor_diameter', 5),
            efficiency=mat_params.get('wind_efficiency', 0.4)
        )
        self.hydro = HydroGenerator(
            efficiency=mat_params.get('hydro_efficiency', 0.7),
            head=mat_params.get('hydro_head', 10)
        )
        self.battery = BatteryStorage(
            capacity=mat_params.get('battery_capacity', 10000),
            charge_rate=mat_params.get('battery_charge_rate', 2000),
            discharge_rate=mat_params.get('battery_discharge_rate', 2000)
        )
        self.backup_gen = BackupGenerator(
            max_output=mat_params.get('backup_max_output', 5000)
        )
        self.town_hours = []
        self.town_total_generation = []
        self.town_total_load = []
        self.town_battery_energy = []
        self.town_total_unmet = []
        self.town_backup_generation = []
        self.final_generation = {"solar": 0, "wind": 0, "hydro": 0}
        self.log = []

    # Standard (non-real-time) simulation remains unchanged…
    def simulate_town(self, houses, total_hours=24, load_scale=1.0):
        self.town_hours = []
        self.town_total_generation = []
        self.town_total_load = []
        self.town_battery_energy = []
        self.town_total_unmet = []
        self.town_backup_generation = []
        house_consumption = {house.name: 0 for house in houses}

        for hour in range(total_hours):
            self.town_hours.append(hour)
            peak_factor = 1.5 if 7 <= hour < 17 else 0.5

            solar_online = np.random.rand() > 0.05
            wind_online = np.random.rand() > 0.05
            hydro_online = np.random.rand() > 0.05

            weather = self.weather_predictor.get_weather(hour)
            solar_energy = (self.solar.generate_power(weather) if solar_online else 0) * peak_factor
            wind_energy = (self.wind.generate_power(weather) if wind_online else 0) * peak_factor
            hydro_energy = (self.hydro.generate_power(weather) if hydro_online else 0) * peak_factor

            self.final_generation["solar"] = max(self.final_generation["solar"], solar_energy)
            self.final_generation["wind"] = max(self.final_generation["wind"], wind_energy)
            self.final_generation["hydro"] = max(self.final_generation["hydro"], hydro_energy)

            if solar_energy < 300:
                self.log.append(f"Hour {hour:02d}: Warning - Low Solar ({solar_energy:.1f} Wh)")
            if wind_energy < 200:
                self.log.append(f"Hour {hour:02d}: Warning - Low Wind ({wind_energy:.1f} Wh)")
            if hydro_energy < 250:
                self.log.append(f"Hour {hour:02d}: Warning - Low Hydro ({hydro_energy:.1f} Wh)")

            R = solar_energy + wind_energy + hydro_energy
            total_load = 0
            for house in houses:
                load = house.get_load(hour) * load_scale
                total_load += load
                house_consumption[house.name] += load

            if R >= total_load:
                surplus = R - total_load
                self.battery.charge(surplus)
                unmet = 0
            else:
                deficit = total_load - R
                battery_used = self.battery.discharge(deficit)
                unmet = total_load - (R + battery_used)

            backup_energy = 0
            if unmet > 0:
                backup_energy = self.backup_gen.generate_power(unmet)
                self.backup_gen.active = True
                unmet = total_load - (R + battery_used + backup_energy)
            elif unmet <= 0 and self.backup_gen.active:
                available_backup_capacity = self.backup_gen.max_output
                needed_charge = max(0, 5000 - self.battery.current_energy)
                if needed_charge > 0:
                    self.battery.charge(min(available_backup_capacity, needed_charge))
                if self.battery.current_energy >= 5000:
                    self.backup_gen.reset()
            self.town_backup_generation.append(backup_energy)
            self.town_total_generation.append(R)
            self.town_total_load.append(total_load)
            self.town_battery_energy.append(self.battery.current_energy)
            self.town_total_unmet.append(unmet)

            self.log.append(f"Hour {hour:02d}: Gen = {R:.1f} Wh, Load = {total_load:.1f} Wh, Battery = {self.battery.current_energy:.1f} Wh, Backup = {backup_energy:.1f} Wh, Unmet = {unmet:.1f} Wh")
        return house_consumption
    def get_town_results_figure(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        axs[0].plot(self.town_hours, self.town_total_generation, label="Total Generation", marker='o')
        axs[0].plot(self.town_hours, self.town_total_load, label="Total Load", marker='o')
        axs[0].set_ylabel("Energy (Wh)")
        axs[0].set_title("Town Energy: Generation vs. Load")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.town_hours, self.town_battery_energy, label="Battery Energy", color="green", marker='o')
        axs[1].set_ylabel("Battery Energy (Wh)")
        axs[1].set_title("Battery Storage Level")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.town_hours, self.town_backup_generation, label="Backup Generation", color="orange", marker='o')
        axs[2].set_ylabel("Backup Generation (Wh)")
        axs[2].set_title("Backup Generator Output")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(self.town_hours, self.town_total_unmet, label="Total Unmet Load", color="red", marker='o')
        axs[3].set_xlabel("Hour")
        axs[3].set_ylabel("Unmet Load (Wh)")
        axs[3].set_title("Total Unmet Load")
        axs[3].legend()
        axs[3].grid(True)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()
        return plot_url

    def get_gauge_charts(self, sim_hours):
        solar_max = 2000
        wind_max = 3000
        hydro_max = 40000
        backup_max_total = 5000 * sim_hours
        final_gen = self.final_generation
        solar_percent = (final_gen["solar"] / solar_max) * 100
        wind_percent = (final_gen["wind"] / wind_max) * 100
        hydro_percent = (final_gen["hydro"] / hydro_max) * 100
        backup_total = sum(self.town_backup_generation)
        backup_percent = (backup_total / backup_max_total) * 100
        fig_solar = go.Figure(go.Indicator(
            mode="gauge+number",
            value=solar_percent,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Solar Generation (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        fig_wind = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wind_percent,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Wind Generation (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        fig_hydro = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hydro_percent,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Hydro Generation (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        fig_backup = go.Figure(go.Indicator(
            mode="gauge+number",
            value=backup_percent,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Backup Generation (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        return (fig_solar.to_html(full_html=False),
                fig_wind.to_html(full_html=False),
                fig_hydro.to_html(full_html=False),
                fig_backup.to_html(full_html=False))

    # New method for dynamic simulation that updates one continuously refreshed image.
    # --- In your EnergyManagementSystem class ---

    def simulate_town_dynamic(self, houses, total_hours=24, load_scale=1.0, delay=1):
        """
        Runs the town simulation dynamically, updating the graph each simulation hour.
        """
        # Clear previous simulation data
        self.town_hours = []
        self.town_total_generation = []
        self.town_total_load = []
        self.town_battery_energy = []
        self.town_total_unmet = []
        self.town_backup_generation = []
        self.log = []

        for hour in range(total_hours):
            self.town_hours.append(hour)
            # Use a peak factor for hours between 7 and 16
            peak_factor = 1.5 if 7 <= hour < 17 else 0.5

            # Each renewable source may be offline (5% chance)
            solar_online = np.random.rand() > 0.05
            wind_online = np.random.rand() > 0.05
            hydro_online = np.random.rand() > 0.05

            weather = self.weather_predictor.get_weather(hour)
            solar_energy = (self.solar.generate_power(weather) if solar_online else 0) * peak_factor
            wind_energy = (self.wind.generate_power(weather) if wind_online else 0) * peak_factor
            hydro_energy = (self.hydro.generate_power(weather) if hydro_online else 0) * peak_factor

            # Update final generation records (if needed)
            self.final_generation["solar"] = max(self.final_generation.get("solar", 0), solar_energy)
            self.final_generation["wind"] = max(self.final_generation.get("wind", 0), wind_energy)
            self.final_generation["hydro"] = max(self.final_generation.get("hydro", 0), hydro_energy)

            # Calculate total load (each house’s load multiplied by load_scale)
            total_load = 0
            for house in houses:
                load = house.get_load(hour) * load_scale
                total_load += load

            R = solar_energy + wind_energy + hydro_energy

            # Simple battery & backup management
            if R >= total_load:
                surplus = R - total_load
                self.battery.charge(surplus)
                unmet = 0
            else:
                deficit = total_load - R
                battery_used = self.battery.discharge(deficit)
                unmet = total_load - (R + battery_used)

            backup_energy = 0
            if unmet > 0:
                backup_energy = self.backup_gen.generate_power(unmet)
                unmet = max(0, total_load - (R + battery_used + backup_energy))
            elif unmet <= 0 and self.backup_gen.active:
                available_backup_capacity = self.backup_gen.max_output
                needed_charge = max(0, 5000 - self.battery.current_energy)
                if needed_charge > 0:
                    self.battery.charge(min(available_backup_capacity, needed_charge))
                if self.battery.current_energy >= 5000:
                    self.backup_gen.reset()

            self.town_backup_generation.append(backup_energy)
            self.town_total_generation.append(R)
            self.town_total_load.append(total_load)
            self.town_battery_energy.append(self.battery.current_energy)
            self.town_total_unmet.append(unmet)
            self.log.append(
                f"Hour {hour:02d}: Gen = {R:.1f} Wh, Load = {total_load:.1f} Wh, Battery = {self.battery.current_energy:.1f} Wh, Backup = {backup_energy:.1f} Wh, Unmet = {unmet:.1f} Wh")

            # Generate the updated graph image and emit via Socket.IO
            img_base64 = self.generate_graph_image()
            socketio.emit('update_image', {'image': img_base64})
            eventlet.sleep(delay)

    def generate_graph_image(self):
        """
        Generates a matplotlib figure (with the data accumulated so far) and returns a base64-encoded image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.town_hours, self.town_total_generation, label="Total Generation", marker='o', color='blue')
        ax.plot(self.town_hours, self.town_total_load, label="Total Load", marker='s', color='red')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Energy (Wh)")
        ax.set_title("Real-Time Town Simulation: Generation vs Load")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

def simulate_historical_dynamic(num_houses, sim_hours, city, date_str, delay=1):
    # Fetch historical weather data
    hist_weather_data = fetch_weather_data(WEATHERSTACK_API_KEY, city, date_str)
    if not hist_weather_data:
        socketio.emit('update_image', {'image': ''})
        return
    irradiances_hist, wind_speeds_hist, water_flows_hist = hist_weather_data
    hist_houses = [House(f"House_{i+1}", np.random.randint(800, 2000)) for i in range(num_houses)]
    hist_generation_data, hist_load_data, hist_battery_energy = [], [], []

    for hour in range(sim_hours):
        solar_energy = np.random.choice(irradiances_hist)
        wind_energy = np.random.choice(wind_speeds_hist)
        hydro_energy = np.random.choice(water_flows_hist)
        total_generation = solar_energy + wind_energy + hydro_energy
        total_load = sum(h.get_load(hour) for h in hist_houses)
        hist_generation_data.append(total_generation)
        hist_load_data.append(total_load)
        hist_battery_energy.append(5000 + hour * 50)  # simulated battery energy

        # Build the dynamic plot for historical simulation:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(range(hour + 1), hist_generation_data, label="Total Generation", marker="o")
        axs[0].plot(range(hour + 1), hist_load_data, label="Total Load", marker="x")
        axs[0].set_ylabel("Energy (Wh)")
        axs[0].set_title("Historical Simulation: Generation vs Load")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(range(hour + 1), hist_battery_energy, label="Battery Energy", color="green", marker="o")
        axs[1].set_xlabel("Hour")
        axs[1].set_ylabel("Energy (Wh)")
        axs[1].set_title("Battery Energy Over Time")
        axs[1].legend()
        axs[1].grid(True)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        socketio.emit('update_image', {'image': img_base64})
        eventlet.sleep(delay)

# -----------------------------------------------------------------------------
# Flask Routes for Standard Pages
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/town_simulation', methods=['GET', 'POST'])
def town_simulation():
    if request.method == 'POST':
        # Retrieve simulation parameters from the form (including file upload)
        sim_hours = int(request.form.get('sim_hours', 24))
        num_houses = int(request.form.get('num_houses', 10))
        load_scale = float(request.form.get('load_scale', 1.0))
        mat_file = request.files.get('mat_file')
        if not mat_file or mat_file.filename == '':
            flash("No MAT file uploaded", "error")
            return redirect(request.url)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], mat_file.filename)
        mat_file.save(file_path)
        mat_data = loadmat(file_path)
        # Build mat_params from file data (as before)
        mat_params = {
            'weather_max_irradiance': mat_data.get('weather_max_irradiance', [[1000]])[0][0],
            'weather_avg_wind': mat_data.get('weather_avg_wind', [[8]])[0][0],
            'weather_avg_water_flow': mat_data.get('weather_avg_water_flow', [[500]])[0][0],
            'solar_area': mat_data.get('solar_area', [[10]])[0][0],
            'solar_efficiency': mat_data.get('solar_efficiency', [[0.15]])[0][0],
            'wind_rotor_diameter': mat_data.get('wind_rotor_diameter', [[5]])[0][0],
            'wind_efficiency': mat_data.get('wind_efficiency', [[0.4]])[0][0],
            'hydro_efficiency': mat_data.get('hydro_efficiency', [[0.7]])[0][0],
            'hydro_head': mat_data.get('hydro_head', [[10]])[0][0],
            'battery_capacity': mat_data.get('battery_capacity', [[10000]])[0][0],
            'battery_charge_rate': mat_data.get('battery_charge_rate', [[2000]])[0][0],
            'battery_discharge_rate': mat_data.get('battery_discharge_rate', [[2000]])[0][0],
            'backup_max_output': mat_data.get('backup_max_output', [[5000]])[0][0],
            'house_base_load_min': mat_data.get('house_base_load_min', [[800]])[0][0],
            'house_base_load_max': mat_data.get('house_base_load_max', [[2000]])[0][0]
        }
        houses = []
        for i in range(num_houses):
            base_load = np.random.randint(mat_params['house_base_load_min'], mat_params['house_base_load_max'] + 1)
            houses.append(House(f"House_{i+1}", base_load))
        # Create an EnergyManagementSystem instance using mat_params
        sems = EnergyManagementSystem(mat_params)
        # Start the dynamic simulation in a background task
        socketio.start_background_task(sems.simulate_town_dynamic, houses, sim_hours, load_scale, delay=1)
        # Render a dynamic simulation page that contains an image placeholder
        return render_template('town_simulation_dynamic.html')
    return render_template('town_simulation.html')


@app.route('/best_forecast', methods=['GET', 'POST'])
def best_forecast():
    if request.method == 'POST':
        place = request.form.get('place', 'New Delhi')
        lat, lon = get_lat_lon(place)
        if lat is None or lon is None:
            flash(f"Location '{place}' not found.", "error")
            return redirect(request.url)
        irradiance, wind, hydro, timestamps = fetch_hourly_forecast(lat, lon)
        if irradiance is None:
            flash("Failed to fetch forecast data.", "error")
            return redirect(request.url)
        solar_pred, wind_pred, hydro_pred = predict_hourly_energy(irradiance, wind, hydro)
        forecast_data = {
            "solar_forecast": np.array(solar_pred),
            "wind_forecast": np.array(wind_pred),
            "hydro_forecast": np.array(hydro_pred),
            "timestamps": np.array(timestamps)
        }
        forecast_file = os.path.join(app.config['UPLOAD_FOLDER'], "energy_hourly_forecast.mat")
        scipy.io.savemat(forecast_file, forecast_data)
        fig, ax = plt.subplots(figsize=(12, 5))
        hours = list(range(len(solar_pred)))
        ax.plot(hours, solar_pred, marker="o", label="Solar Generation (kWh)", color="orange")
        ax.plot(hours, wind_pred, marker="s", label="Wind Generation (kWh)", color="blue")
        ax.plot(hours, hydro_pred, marker="^", label="Hydro Generation (kWh)", color="green")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Estimated Energy (kWh)")
        ax.set_title("Predicted Renewable Energy for Next 48 Hours")
        ax.legend()
        ax.grid(True)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        forecast_plot = base64.b64encode(buf.getvalue()).decode()
        return render_template('forecast_result.html', forecast_plot=forecast_plot, place=place)
    return render_template('best_forecast.html')



# -----------------------------------------------------------------------------
# Dynamic Simulation Route & Socket.IO Events
# -----------------------------------------------------------------------------
@app.route('/dynamic_simulation')
def dynamic_simulation():
    return render_template('simulation_dynamic.html')

@app.route('/start_dynamic_simulation')
def start_dynamic_simulation():
    # Set up default simulation parameters
    sim_hours = 24
    num_houses = 10
    load_scale = 1.0
    delay = 1  # Update graph every second

    # Create house instances
    houses = [House(f"House_{i + 1}", np.random.randint(800, 2000)) for i in range(num_houses)]

    # Create Energy Management System
    sems = EnergyManagementSystem({})

    # Start the real-time simulation in a background task
    socketio.start_background_task(sems.simulate_town_dynamic, houses, sim_hours, load_scale, delay)

    return "Dynamic simulation started!"




# -----------------------------------------------------------------------------
# Helper Functions for Forecast & Historical Data
# -----------------------------------------------------------------------------
def get_lat_lon(place_name):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={place_name}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(geo_url)
    if response.status_code == 200 and len(response.json()) > 0:
        location = response.json()[0]
        return float(location["lat"]), float(location["lon"])
    else:
        return None, None

def fetch_hourly_forecast(lat, lon):
    forecast_api_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(forecast_api_url)
    if response.status_code == 200:
        data = response.json()
        forecast_list = data.get("list", [])[:48]
        timestamps = [f.get("dt_txt", "Unknown") for f in forecast_list]
        irradiance = [max(0, 1000*(1 - f.get("clouds", {}).get("all", 50)/100)) for f in forecast_list]
        wind_speed = [f.get("wind", {}).get("speed", 5)*3.6 for f in forecast_list]
        hydro_flow = [f.get("rain", {}).get("3h", 0)*100 for f in forecast_list]
        return irradiance, wind_speed, hydro_flow, timestamps
    return None, None, None, None

def predict_hourly_energy(irradiance, wind, hydro):
    solar_efficiency = 0.25
    wind_efficiency = 0.5
    hydro_efficiency = 0.9
    solar_generation = [solar_efficiency * 50 * i for i in irradiance]
    wind_generation = [wind_efficiency * 12 * (w ** 3) for w in wind]
    hydro_generation = [(h/1000)*9.81*10*hydro_efficiency*1000 for h in hydro]
    return solar_generation, wind_generation, hydro_generation

def fetch_weather_data(api_key, city, date):
    weather_api_url = f"http://api.weatherstack.com/historical?access_key={api_key}&query={city}&historical_date={date}"
    response = requests.get(weather_api_url)
    if response.status_code == 200:
        data = response.json()
        hourly_data = data.get("forecast", {}).get("forecastday", [{}])[0].get("hour", [])
        irradiances, wind_speeds, water_flows = [], [], []
        for hour in hourly_data:
            cloud = hour.get("cloud", 50)
            irradiance = max(0, 1000*(1 - cloud/100))
            irradiances.append(irradiance)
            wind_speeds.append(hour.get("wind_kph", 20))
            water_flows.append(hour.get("precip_mm", 0)*100)
        return irradiances, wind_speeds, water_flows
    else:
        return None

@app.route('/historical_simulation', methods=['GET', 'POST'])
def historical_simulation():
    if request.method == 'POST':
        num_houses = int(request.form.get('num_houses', 10))
        sim_hours = int(request.form.get('sim_hours', 24))
        city = request.form.get('city', 'New Delhi')
        date_str = request.form.get('date')
        if not date_str:
            flash("Please select a date.", "error")
            return redirect(request.url)
        # Use the new Tomorrow.io API function:
        hist_weather_data = fetch_historical_weather_tomorrow(city, date_str)
        if hist_weather_data:
            irradiances_hist, wind_speeds_hist, water_flows_hist = hist_weather_data
        else:
            flash("Failed to fetch historical weather data from Tomorrow.io.", "error")
            return redirect(request.url)
        # Start the dynamic historical simulation as a background task:
        socketio.start_background_task(simulate_historical_dynamic, num_houses, sim_hours, city, date_str, 1)
        return render_template('historical_dynamic.html')
    return render_template('historical_simulation.html')


# -----------------------------------------------------------------------------
# Socket.IO Event (client listens for 'update_image')
# -----------------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    print("Client connected.")

# -----------------------------------------------------------------------------
# Run the App
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, debug=True)
