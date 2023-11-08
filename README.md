# tomorrow-weather-predict
<br>This is a python - spark academic project to predict the weather for tomorrow given today's date, latitude, longtitude, elevation, and temperature. 
<br>To run this program:
<br>spark-submit weather_train.py tmax-1 weather-model
<br>OR
<br>spark-submit weather_train.py tmax-2 weather-model
<br>THEN 
<br>spark-submit weather_tomorrow.py weather-model
