from django.urls import path

from .views import similar_trades, trading_leaderboard

urlpatterns = [
    path("", trading_leaderboard, name="trading-leaderboard"),
    path("leaderboard/", trading_leaderboard, name="trading-leaderboard"),
    path("similar-trades/", similar_trades, name="trading-similar-trades"),
]
