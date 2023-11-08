# tomorrow-weather-predict
<br>this is a python - spark academic project to predict the weather for tomorrow given today's date, latitude, longtitude, elevation, and temperature. 
<br>to run this program:
<br>spark-submit weather_train.py tmax-1 weather-model
<br>or
<br>spark-submit weather_train.py tmax-2 weather-model
<br>then 
<br>spark-submit weather_tomorrow.py weather-model
