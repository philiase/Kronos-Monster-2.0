from .market_structure import (
    LiquidityLevel,
    MarketStructure,
    PriceLevel,
    StructureConfig,
    analyze_market_structure,
)
from .scenarios import ScenarioConfig, ScenarioScore, score_forecast_scenario, select_best_scenario
from .signals import SignalConfig, TradeSignal, analyze_forecast, summarize_signal


class _LazyKronosClass:
    def __init__(self, class_name):
        self.class_name = class_name

    def _load(self):
        try:
            from . import kronos
        except Exception as exc:
            raise ImportError(
                "Kronos model dependencies could not be imported. "
                "Install a working torch environment to use model inference."
            ) from exc
        return getattr(kronos, self.class_name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


KronosTokenizer = _LazyKronosClass("KronosTokenizer")
Kronos = _LazyKronosClass("Kronos")
KronosPredictor = _LazyKronosClass("KronosPredictor")

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


