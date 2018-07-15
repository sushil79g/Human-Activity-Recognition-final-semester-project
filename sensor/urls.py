from django.urls import path
from . import views

app_name='sensor'

urlpatterns = [
    path('', views.AccelerometerView.as_view(), name="accelerometer_view"),
]