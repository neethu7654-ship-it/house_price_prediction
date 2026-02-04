from django.urls import path
from . import views

urlpatterns = [
    path('', views.price_calculator, name='price_calculator'),
]