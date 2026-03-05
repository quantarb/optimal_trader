from django.db import models


class Symbol(models.Model):
    symbol = models.CharField(max_length=32, unique=True, db_index=True)
    company_name = models.CharField(max_length=255, blank=True, default="")
    exchange = models.CharField(max_length=64, blank=True, default="")
    country = models.CharField(max_length=8, blank=True, default="")
    sector = models.CharField(max_length=128, blank=True, default="")
    industry = models.CharField(max_length=255, blank=True, default="")

    market_cap = models.FloatField(null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    beta = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    dividend = models.FloatField(null=True, blank=True)
    dividend_yield = models.FloatField(null=True, blank=True)

    payload = models.JSONField(default=dict, blank=True)
    historical_date_ranges = models.JSONField(default=dict, blank=True)
    last_date_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["symbol"]

    def __str__(self) -> str:
        return self.symbol


class Industry(models.Model):
    name = models.CharField(max_length=255, unique=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Sector(models.Model):
    name = models.CharField(max_length=255, unique=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Exchange(models.Model):
    code = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=255, blank=True, default="")
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.code


class Country(models.Model):
    code = models.CharField(max_length=16, unique=True)
    name = models.CharField(max_length=255, blank=True, default="")
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.code


class EconomicIndicatorSeries(models.Model):
    code = models.CharField(max_length=128, unique=True, db_index=True)
    display_name = models.CharField(max_length=255, blank=True, default="")
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    min_date = models.DateField(null=True, blank=True)
    max_date = models.DateField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.code


class EconomicIndicatorObservation(models.Model):
    series = models.ForeignKey(EconomicIndicatorSeries, on_delete=models.CASCADE, related_name="observations")
    observation_date = models.DateField()
    value = models.FloatField()
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("series", "observation_date"),)
        indexes = [
            models.Index(fields=["series", "-observation_date"]),
        ]
        ordering = ["series__code", "-observation_date"]

    def __str__(self) -> str:
        return f"{self.series.code}:{self.observation_date.isoformat()}"


class TreasuryRateSeries(models.Model):
    code = models.CharField(max_length=128, unique=True, db_index=True)
    display_name = models.CharField(max_length=255, blank=True, default="")
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    min_date = models.DateField(null=True, blank=True)
    max_date = models.DateField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.code


class TreasuryRateObservation(models.Model):
    series = models.ForeignKey(TreasuryRateSeries, on_delete=models.CASCADE, related_name="observations")
    observation_date = models.DateField()
    value = models.FloatField()
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("series", "observation_date"),)
        indexes = [
            models.Index(fields=["series", "-observation_date"]),
        ]
        ordering = ["series__code", "-observation_date"]

    def __str__(self) -> str:
        return f"{self.series.code}:{self.observation_date.isoformat()}"


class MacroSeries(models.Model):
    code = models.CharField(max_length=128, unique=True, db_index=True)
    display_name = models.CharField(max_length=255, blank=True, default="")
    category = models.CharField(max_length=32, blank=True, default="economic")
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    min_date = models.DateField(null=True, blank=True)
    max_date = models.DateField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.code


class MacroObservation(models.Model):
    series = models.ForeignKey(MacroSeries, on_delete=models.CASCADE, related_name="observations")
    observation_date = models.DateField()
    value = models.FloatField()
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("series", "observation_date"),)
        indexes = [
            models.Index(fields=["series", "-observation_date"]),
        ]
        ordering = ["series__code", "-observation_date"]

    def __str__(self) -> str:
        return f"{self.series.code}:{self.observation_date.isoformat()}"


class SymbolSectionState(models.Model):
    KIND_CHOICES = (
        ("snapshot", "snapshot"),
        ("historical", "historical"),
    )

    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name="section_states")
    section_key = models.CharField(max_length=100)
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("symbol", "section_key"),)
        ordering = ["section_key"]

    def __str__(self) -> str:
        return f"{self.symbol.symbol}:{self.section_key}"


class SymbolSectionSnapshot(models.Model):
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name="section_snapshots")
    section_key = models.CharField(max_length=100)
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("symbol", "section_key"),)
        ordering = ["section_key"]

    def __str__(self) -> str:
        return f"{self.symbol.symbol}:{self.section_key}"


class SymbolSectionHistorical(models.Model):
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name="section_history")
    section_key = models.CharField(max_length=100)
    record_key = models.CharField(max_length=64)
    record_date = models.DateField(null=True, blank=True)
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("symbol", "section_key", "record_key"),)
        indexes = [
            models.Index(fields=["symbol", "section_key", "-record_date"]),
        ]
        ordering = ["section_key", "-record_date", "-updated_at"]

    def __str__(self) -> str:
        return f"{self.symbol.symbol}:{self.section_key}:{self.record_key[:8]}"
