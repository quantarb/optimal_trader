from django import forms
from typing import Optional, Sequence
from django.utils import timezone


BOOL_CHOICES = (
    ("", "Any"),
    ("true", "True"),
    ("false", "False"),
)

UNIVERSE_DATA_SOURCE_CHOICES = (
    ("api", "FMP API (online)"),
    ("db", "Django DB (local cache)"),
)

EXCHANGE_CHOICES = (
    ("", "Any"),
    ("NASDAQ", "NASDAQ"),
    ("NYSE", "NYSE"),
    ("AMEX", "AMEX"),
    ("TSX", "TSX"),
    ("TSXV", "TSXV"),
    ("LSE", "LSE"),
    ("EURONEXT", "EURONEXT"),
    ("XETRA", "XETRA"),
    ("HKSE", "HKSE"),
    ("SSE", "SSE"),
    ("SZSE", "SZSE"),
    ("TSE", "TSE"),
    ("ASX", "ASX"),
    ("NSE", "NSE"),
    ("BSE", "BSE"),
)

SECTOR_CHOICES = (
    ("", "Any"),
    ("Communication Services", "Communication Services"),
    ("Consumer Cyclical", "Consumer Cyclical"),
    ("Consumer Defensive", "Consumer Defensive"),
    ("Energy", "Energy"),
    ("Financial Services", "Financial Services"),
    ("Healthcare", "Healthcare"),
    ("Industrials", "Industrials"),
    ("Basic Materials", "Basic Materials"),
    ("Real Estate", "Real Estate"),
    ("Technology", "Technology"),
    ("Utilities", "Utilities"),
)

INDUSTRY_CHOICES = (
    ("", "Any"),
    ("Software - Infrastructure", "Software - Infrastructure"),
    ("Software - Application", "Software - Application"),
    ("Semiconductors", "Semiconductors"),
    ("Internet Content & Information", "Internet Content & Information"),
    ("Consumer Electronics", "Consumer Electronics"),
    ("Auto Manufacturers", "Auto Manufacturers"),
    ("Banks - Diversified", "Banks - Diversified"),
    ("Credit Services", "Credit Services"),
    ("Insurance - Diversified", "Insurance - Diversified"),
    ("Asset Management", "Asset Management"),
    ("Drug Manufacturers - General", "Drug Manufacturers - General"),
    ("Biotechnology", "Biotechnology"),
    ("Medical Devices", "Medical Devices"),
    ("Oil & Gas Integrated", "Oil & Gas Integrated"),
    ("Oil & Gas E&P", "Oil & Gas E&P"),
    ("Aerospace & Defense", "Aerospace & Defense"),
    ("Specialty Retail", "Specialty Retail"),
    ("Discount Stores", "Discount Stores"),
    ("Beverages - Non-Alcoholic", "Beverages - Non-Alcoholic"),
    ("Restaurants", "Restaurants"),
    ("REIT - Industrial", "REIT - Industrial"),
    ("REIT - Retail", "REIT - Retail"),
    ("Utilities - Regulated Electric", "Utilities - Regulated Electric"),
)

COUNTRY_CHOICES = (
    ("", "Any"),
    ("US", "United States"),
    ("CA", "Canada"),
    ("GB", "United Kingdom"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("NL", "Netherlands"),
    ("CH", "Switzerland"),
    ("SE", "Sweden"),
    ("ES", "Spain"),
    ("IT", "Italy"),
    ("JP", "Japan"),
    ("CN", "China"),
    ("HK", "Hong Kong"),
    ("KR", "South Korea"),
    ("TW", "Taiwan"),
    ("IN", "India"),
    ("SG", "Singapore"),
    ("AU", "Australia"),
    ("BR", "Brazil"),
)

ECONOMIC_INDICATOR_SERIES = (
    "GDP",
    "realGDP",
    "nominalPotentialGDP",
    "realGDPPerCapita",
    "nominalGDP",
    "federalFunds",
    "CPI",
    "inflationRate",
    "inflation",
    "retailSales",
    "consumerSentiment",
    "durableGoods",
    "unemploymentRate",
    "totalNonfarmPayroll",
    "initialClaims",
    "industrialProductionTotalIndex",
    "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
    "totalVehicleSales",
    "retailMoneyFunds",
    "smoothedUSRecessionProbabilities",
    "3MonthOr90DayRatesAndYieldsCertificatesOfDeposit",
    "commercialBankInterestRateOnCreditCardPlansAllAccounts",
    "30YearFixedRateMortgageAverage",
    "15YearFixedRateMortgageAverage",
    "tradeBalanceGoodsAndServices",
)

MACRO_SERIES_CHOICES = tuple((series, series) for series in ECONOMIC_INDICATOR_SERIES)


def _find_us_country_value(choices: Sequence[tuple[str, str]]) -> str:
    for value, _label in choices:
        if str(value).strip().upper() == "US":
            return str(value)
    for value, label in choices:
        v = str(value).strip().lower()
        l = str(label).strip().lower()
        if v in {"united states", "usa"} or "united states" in l:
            return str(value)
    return "US"


class UniverseScreenerForm(forms.Form):
    data_source = forms.ChoiceField(required=False, choices=UNIVERSE_DATA_SOURCE_CHOICES, initial="db")
    limit = forms.IntegerField(required=False, min_value=1, initial=10000)
    marketCapMoreThan = forms.FloatField(required=False, label="Market Cap Min (Millions)", initial=5000)
    marketCapLowerThan = forms.FloatField(required=False, label="Market Cap Max (Millions)")
    sector = forms.ChoiceField(required=False, choices=SECTOR_CHOICES)
    industry = forms.ChoiceField(required=False, choices=INDUSTRY_CHOICES)
    betaMoreThan = forms.FloatField(required=False)
    betaLowerThan = forms.FloatField(required=False)
    priceMoreThan = forms.FloatField(required=False)
    priceLowerThan = forms.FloatField(required=False)
    dividendMoreThan = forms.FloatField(required=False)
    dividendLowerThan = forms.FloatField(required=False)
    volumeMoreThan = forms.FloatField(required=False, label="Volume Min (Millions)")
    volumeLowerThan = forms.FloatField(required=False, label="Volume Max (Millions)")
    exchange = forms.MultipleChoiceField(
        required=False,
        choices=EXCHANGE_CHOICES[1:],
        widget=forms.SelectMultiple(attrs={"size": 8}),
    )
    country = forms.ChoiceField(required=False, choices=COUNTRY_CHOICES, initial="US")
    isEtf = forms.ChoiceField(required=False, choices=BOOL_CHOICES)
    isFund = forms.ChoiceField(required=False, choices=BOOL_CHOICES, initial="false")
    isActivelyTrading = forms.ChoiceField(required=False, choices=BOOL_CHOICES)
    includeAllShareClasses = forms.ChoiceField(required=False, choices=BOOL_CHOICES, initial="false")

    def __init__(
        self,
        *args,
        sector_choices: Optional[Sequence[tuple[str, str]]] = None,
        industry_choices: Optional[Sequence[tuple[str, str]]] = None,
        exchange_choices: Optional[Sequence[tuple[str, str]]] = None,
        country_choices: Optional[Sequence[tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sector_choices:
            self.fields["sector"].choices = list(sector_choices)
        if industry_choices:
            self.fields["industry"].choices = list(industry_choices)
        if exchange_choices:
            self.fields["exchange"].choices = list(exchange_choices)
        if country_choices:
            self.fields["country"].choices = list(country_choices)
        if not self.is_bound:
            country_field = self.fields["country"]
            country_values = {str(v) for v, _ in country_field.choices}
            existing = self.initial.get("country", country_field.initial)
            if str(existing) not in country_values:
                fallback_country = _find_us_country_value(country_field.choices)
                self.initial["country"] = fallback_country
                country_field.initial = fallback_country


class EconomicIndicatorsForm(forms.Form):
    start_date = forms.DateField(required=True)
    end_date = forms.DateField(required=True)
    economic_series = forms.MultipleChoiceField(
        required=False,
        choices=MACRO_SERIES_CHOICES,
        widget=forms.SelectMultiple(attrs={"size": 6}),
        help_text="Select one or more macro series.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_bound:
            today = timezone.now().date()
            self.fields["end_date"].initial = today
            self.fields["start_date"].initial = today.replace(year=1900, month=1, day=1)
            self.fields["economic_series"].initial = ["GDP", "CPI", "unemploymentRate", "federalFunds"]


class TreasuryRatesForm(forms.Form):
    start_date = forms.DateField(required=True)
    end_date = forms.DateField(required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_bound:
            today = timezone.now().date()
            self.fields["end_date"].initial = today
            self.fields["start_date"].initial = today.replace(year=1900, month=1, day=1)
