# tomorrow-weather-predict
this is a python - spark academic project to predict the weather for tomorrow given today's date, latitude, longtitude, elevation, and temperature. 
to run this program:
spark-submit weather_train.py tmax-1 weather-model
or
spark-submit weather_train.py tmax-2 weather-model
then 
spark-submit weather_tomorrow.py weather-model
