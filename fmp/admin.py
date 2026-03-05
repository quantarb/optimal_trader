from django.contrib import admin

from .models import (
    Country,
    EconomicIndicatorObservation,
    EconomicIndicatorSeries,
    Exchange,
    Industry,
    MacroObservation,
    MacroSeries,
    Sector,
    Symbol,
    SymbolSectionHistorical,
    SymbolSectionSnapshot,
    SymbolSectionState,
    TreasuryRateObservation,
    TreasuryRateSeries,
)


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = (
        "symbol",
        "company_name",
        "exchange",
        "country",
        "sector",
        "industry",
        "price",
        "market_cap",
        "last_date_updated",
    )
    list_filter = ("exchange", "country", "sector", "industry")
    search_fields = ("symbol", "company_name")
    ordering = ("symbol",)


@admin.register(Industry)
class IndustryAdmin(admin.ModelAdmin):
    list_display = ("name", "last_updated")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Sector)
class SectorAdmin(admin.ModelAdmin):
    list_display = ("name", "last_updated")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Exchange)
class ExchangeAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "last_updated")
    search_fields = ("code", "name")
    ordering = ("code",)


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "last_updated")
    search_fields = ("code", "name")
    ordering = ("code",)


@admin.register(EconomicIndicatorSeries)
class EconomicIndicatorSeriesAdmin(admin.ModelAdmin):
    list_display = ("code", "display_name", "min_date", "max_date", "last_fetched_at", "last_updated")
    search_fields = ("code", "display_name")
    ordering = ("code",)


@admin.register(EconomicIndicatorObservation)
class EconomicIndicatorObservationAdmin(admin.ModelAdmin):
    list_display = ("series", "observation_date", "value", "updated_at")
    list_filter = ("series",)
    search_fields = ("series__code",)
    ordering = ("series__code", "-observation_date")


@admin.register(TreasuryRateSeries)
class TreasuryRateSeriesAdmin(admin.ModelAdmin):
    list_display = ("code", "display_name", "min_date", "max_date", "last_fetched_at", "last_updated")
    search_fields = ("code", "display_name")
    ordering = ("code",)


@admin.register(TreasuryRateObservation)
class TreasuryRateObservationAdmin(admin.ModelAdmin):
    list_display = ("series", "observation_date", "value", "updated_at")
    list_filter = ("series",)
    search_fields = ("series__code",)
    ordering = ("series__code", "-observation_date")


@admin.register(MacroSeries)
class MacroSeriesAdmin(admin.ModelAdmin):
    list_display = ("code", "display_name", "category", "min_date", "max_date", "last_fetched_at", "last_updated")
    list_filter = ("category",)
    search_fields = ("code", "display_name")
    ordering = ("code",)


@admin.register(MacroObservation)
class MacroObservationAdmin(admin.ModelAdmin):
    list_display = ("series", "observation_date", "value", "updated_at")
    list_filter = ("series",)
    search_fields = ("series__code",)
    ordering = ("series__code", "-observation_date")


@admin.register(SymbolSectionState)
class SymbolSectionStateAdmin(admin.ModelAdmin):
    list_display = ("symbol", "section_key", "kind", "last_fetched_at", "updated_at")
    list_filter = ("kind", "section_key")
    search_fields = ("symbol__symbol", "section_key")
    ordering = ("symbol__symbol", "section_key")


@admin.register(SymbolSectionSnapshot)
class SymbolSectionSnapshotAdmin(admin.ModelAdmin):
    list_display = ("symbol", "section_key", "updated_at")
    list_filter = ("section_key",)
    search_fields = ("symbol__symbol", "section_key")
    ordering = ("symbol__symbol", "section_key")


@admin.register(SymbolSectionHistorical)
class SymbolSectionHistoricalAdmin(admin.ModelAdmin):
    list_display = ("symbol", "section_key", "record_key", "record_date", "updated_at")
    list_filter = ("section_key",)
    search_fields = ("symbol__symbol", "section_key", "record_key")
    ordering = ("symbol__symbol", "section_key", "-record_date")
