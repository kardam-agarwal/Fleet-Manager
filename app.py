from flask import Flask,request, render_template
import numpy as np
import pickle

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

# Index route
@app.route('/')
def hello_world():
    return render_template("index.html")

# Login Page route
@app.route('/login')
def login():
    return render_template("login.html")

# Register Page route
@app.route('/register')
def register():
    return render_template("register.html")

# Driver Performance route
@app.route('/driverperf')
def driverperf():
    return render_template("driverPerformance.html")

# Driver Score  route
@app.route('/driverscore')
def driverscore():
    return render_template("DriverScoreboard.html")

# Fleet Mgmt  route
@app.route('/fleetcost')
def fleetcost():
    return render_template("FleetCostManagement.html")

# Fuel Cost route
@app.route('/fuel')
def fuel():
    return render_template("fuel.html")

# Geo Tracking route
@app.route('/geotrack')
def geotrack():
    return render_template("GeoTracking.html")

# Maintenance Status route
@app.route('/maintenance')
def maintenance():
    return render_template("MaintenanceStatus.html")

# Vehicle Health route
@app.route('/vehihealth')
def vehihealth():
    return render_template("VehicleHealthStatus.html")

# Vehicle Inspection Status route
@app.route('/vehiinspec')
def vehiinspec():
    return render_template("VehicleInspectionStatus.html")

# Vehicle Stats route
@app.route('/vehistat')
def vehistat():
    return render_template("VehicleStats.html")

# Driver Score Prediction
@app.route('/predict',methods=['POST','GET'])
def predict():
    #field_names = ["Vehicle Speed (Kmph)", "Vehicle Direction (degrees)", "Vehicle Overtaken", "Licence", "Collision", "Parking Accuracy", "Signal Jumps", "Drunk (mg)"]
    # Field values taken from form
    veh_speed = request.form['veh_speed']
    veh_dir = request.form['veh_dir']
    veh_ovtkn = request.form['veh_ovtkn']
    lic = request.form['lic']
    col = request.form['col']
    park_acc = request.form['park_acc']
    sig_jump = request.form['sig_jump']
    drunk = request.form['drunk']

    # Values array
    values = [veh_speed, veh_dir, veh_ovtkn, lic, col, park_acc, sig_jump, drunk]

    # Numpy array of field values
    field_values = np.array(values).reshape(1, -1)
    
    # Predicted scored from model
    score = model.predict(field_values)

    # Final predicted score out of 100
    output = int((score*100)/80)

    return render_template('DriverScoreboard.html',pred='Driver Score: {}/100'.format(output), params='Parameters for driver score: {}'.format(values))


if __name__ == '__main__':
    app.run(port=5000, debug=False)
