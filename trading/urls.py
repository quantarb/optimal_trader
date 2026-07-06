from django.urls import path

from .views import trading_leaderboard

urlpatterns = [
    path("", trading_leaderboard, name="trading-leaderboard"),
    path("leaderboard/", trading_leaderboard, name="trading-leaderboard"),
]
