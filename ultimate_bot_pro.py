import os
import sys
import time
import math
import json
import uuid
import random
import threading
import sqlite3
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional libraries
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available. Running in simulation mode.")

try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS = True
except ImportError:
    NUMPY_PANDAS = False
    print("NumPy or Pandas not available. Some features may be limited.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using simplified indicators.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. ML features disabled.")

# Dashboard libs (optional)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Tkinter not available. Dashboard disabled.")


class Config:
    MODE = "sim"  # 'sim' or 'live'
    SYMBOLS = ["XAUUSD", "BTCUSD", "US30", "DJI"]

    # MT5 timeframe map
    TF_MAP = {
        "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    }

    # Two-timeframe architecture
    PRIMARY_TF = "H1"   # Trend analysis
    ENTRY_TF = "M15"    # Entry signals

    GOLDEN_ZONE = (0.5, 0.705)

    DB_FILE = "/workspace/ultimate_bot_pro.db"
    LOG_FILE = "/workspace/ultimate_bot_pro.log"
    CACHE_TTL_SEC = 15

    RISK = {
        "low_risk_pct": 1.0,
        "high_risk_pct": 2.0,
        "rr_ratio": 2.0,
        "max_daily_loss_pct": 2.9,
        "max_open_trades": 4,
    }

    BROKER_RULES = {
        "min_lot": 0.01,
        "max_lot": 100.0,
        "lot_step": 0.01,
    }

    BACKTEST = {
        "slippage_pct": 0.0005,
        "spread_map": {"XAUUSD": 0.3, "BTCUSD": 5.0, "US30": 1.0, "DJI": 1.0},
    }

    OPTIMIZER = {
        "ga_pop": 20,
        "ga_gens": 20,
        "random_budget": 50,
        "parallel_workers": 4,
    }

    ML = {
        "retrain_interval_hrs": 24,
        "model_file": "/workspace/ml_recommender.pkl",
    }

    WATCHDOG_INTERVAL_SEC = 10

    SIM = {
        "start_balance": 10000.0,
        "volatility": 0.002,
    }

    THREAD_POOL = {"max_workers": 4}

    # Dashboard related
    USE_DASHBOARD = False
    DASHBOARD = {
        "sessions": ["Asia", "London", "New York"],
        "tf_combos": [("H1", "M15"), ("M15", "M5"), ("M5", "M1")],
        "modules": [
            "RSI Divergence",
            "Candlestick Patterns",
            "Fibonacci Golden Zone",
            "BOS / CHoCH",
            "News Filter",
            "Session Filter",
            "Spread Filter",
            "Volume Filter",
            "Liquidity Zones",
            "Martingale / Scale In",
            "Partial Exit",
            "Drawdown Protector",
            "Daily Risk Controller",
            "Multi-Symbol Orchestrator",
        ],
    }

    # Dev/testing knobs
    LOOP_SLEEP_SEC = 60
    ONE_SHOT_ENV = "ONE_SHOT"  # If set in env, run just one cycle and exit


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ultimate_bot_pro")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Config.LOG_FILE)
        fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fh_formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(ch_formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


logger = setup_logger()


def log(msg: str, level: str = "info") -> None:
    getattr(logger, level)(msg)


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, fn: Callable[[Any], None]) -> None:
        with self._lock:
            if topic not in self._subs:
                self._subs[topic] = []
            self._subs[topic].append(fn)

    def publish(self, topic: str, payload: Any) -> None:
        handlers: List[Callable[[Any], None]] = []
        with self._lock:
            handlers = self._subs.get(topic, [])[:]
        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                log(f"خطا در پردازش رویداد {topic}: {e}\n{traceback.format_exc()}", "error")


event_bus = EventBus()


class TradeMemory:
    def __init__(self, db_file: str = Config.DB_FILE) -> None:
        self.db = db_file
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                ts TEXT,
                symbol TEXT,
                side TEXT,
                entry REAL,
                exit REAL,
                sl REAL,
                tp REAL,
                volume REAL,
                pnl REAL,
                meta TEXT
            )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                ts TEXT,
                symbol TEXT,
                decision TEXT,
                score REAL,
                details TEXT
            )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY,
                ts TEXT,
                symbol TEXT,
                features TEXT,
                label INTEGER
            )"""
            )
            conn.commit()
        log("پایگاه داده حافظه معاملات راه‌اندازی شد", "info")

    def record_trade(self, rec: Dict[str, Any]) -> None:
        with self._lock, sqlite3.connect(self.db) as conn:
            cur = conn.cursor()
            trade_id = rec.get("id", str(uuid.uuid4()))
            timestamp = rec.get("ts", datetime.utcnow().isoformat())
            cur.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trade_id,
                    timestamp,
                    rec["symbol"],
                    rec["side"],
                    rec.get("entry"),
                    rec.get("exit"),
                    rec.get("sl"),
                    rec.get("tp"),
                    rec.get("volume"),
                    rec.get("pnl"),
                    json.dumps(rec.get("meta", {})),
                ),
            )
            conn.commit()
        log(f"معامله ثبت شد: {rec['symbol']} {rec['side']} سود/زیان={rec.get('pnl')}", "info")

    def record_signal(self, symbol: str, decision: str, score: float, details: Dict[str, Any]) -> None:
        with self._lock, sqlite3.connect(self.db) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO signals VALUES (?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    datetime.utcnow().isoformat(),
                    symbol,
                    decision,
                    float(score),
                    json.dumps(details),
                ),
            )
            conn.commit()

    def query_recent_signals(self, limit: int = 50) -> List[Tuple[str, str, str, str, float]]:
        with sqlite3.connect(self.db) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, ts, symbol, decision, score FROM signals ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()


trade_memory = TradeMemory()


class MT5Adapter:
    def __init__(self, mode: str = "sim") -> None:
        self.mode = mode
        self.sim: Dict[str, Dict[str, Any]] = {}
        self.live = False
        if self.mode == "live" and MT5_AVAILABLE:
            self._init_live()
        else:
            self._init_sim()

    def _init_live(self) -> None:
        try:
            if not mt5.initialize():
                raise ConnectionError("Failed to initialize MT5")
            self.live = True
            log("اتصال به MT5 برقرار شد (حالت زنده)", "info")
        except Exception as e:
            log(f"خطا در اتصال به MT5: {e}. تغییر به حالت شبیه‌سازی", "warning")
            self._init_sim()

    def _init_sim(self) -> None:
        self.live = False
        for symbol in Config.SYMBOLS:
            self.sim[symbol] = {
                "price": 100.0 + random.random() * 100.0,
                "vol": Config.SIM["volatility"],
                "history": [],
            }
        log("حالت شبیه‌سازی فعال شد", "info")

    def shutdown(self) -> None:
        if self.live and MT5_AVAILABLE:
            mt5.shutdown()
            log("اتصال به MT5 قطع شد", "info")

    def get_rates(self, symbol: str, timeframe: str, count: int) -> Optional["pd.DataFrame"]:
        if self.live:
            return self._get_live_rates(symbol, timeframe, count)
        else:
            return self._get_sim_rates(symbol, timeframe, count)

    def _get_live_rates(self, symbol: str, timeframe: str, count: int) -> Optional["pd.DataFrame"]:
        if not NUMPY_PANDAS:
            return None
        tf = Config.TF_MAP.get(timeframe, mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15)
        try:
            bars = mt5.copy_rates_from_pos(symbol, tf, 0, count) if MT5_AVAILABLE else None
            if bars is None or len(bars) == 0:
                return None
            df = pd.DataFrame(bars)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df
        except Exception as e:
            log(f"خطا در دریافت داده‌های زنده: {e}", "error")
            return None

    def _get_sim_rates(self, symbol: str, timeframe: str, count: int) -> "pd.DataFrame":
        if not NUMPY_PANDAS:
            raise RuntimeError("Pandas is required for simulation data generation")
        if symbol not in self.sim:
            self.sim[symbol] = {
                "price": 100.0 + random.random() * 100.0,
                "vol": Config.SIM["volatility"],
                "history": [],
            }
        s = self.sim[symbol]
        rows: List[Dict[str, Any]] = []
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        if timeframe == "M1":
            delta = timedelta(minutes=1)
        elif timeframe == "M5":
            delta = timedelta(minutes=5)
        elif timeframe == "M15":
            delta = timedelta(minutes=15)
        elif timeframe == "H1":
            delta = timedelta(hours=1)
        elif timeframe == "H4":
            delta = timedelta(hours=4)
        else:
            delta = timedelta(minutes=15)
        for i in range(count):
            p = s["price"] * (1 + random.gauss(0, s["vol"]))
            o = p * (1 + random.uniform(-0.0005, 0.0005))
            c = p * (1 + random.uniform(-0.0005, 0.0005))
            h = max(o, c) * (1 + abs(random.uniform(0, 0.0005)))
            l = min(o, c) * (1 - abs(random.uniform(0, 0.0005)))
            ticks = random.randint(1, 100)
            ts = now - (count - i - 1) * delta
            rows.append({
                "time": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "tick_volume": ticks,
            })
            s["history"].append({"time": ts, "open": o, "high": h, "low": l, "close": c})
            s["price"] = s["price"] * (1 + random.gauss(0, s["vol"] * 0.1))
        df = pd.DataFrame(rows)
        return df

    def symbol_info(self, symbol: str):
        if self.live and MT5_AVAILABLE:
            try:
                return mt5.symbol_info(symbol)
            except Exception:
                return None
        else:
            class SymbolInfo:
                def __init__(self) -> None:
                    self.point = 0.01
                    self.spread = Config.BACKTEST["spread_map"].get(symbol, 1.0)
                    self.volume_min = Config.BROKER_RULES["min_lot"]
                    self.volume_max = Config.BROKER_RULES["max_lot"]
                    self.volume_step = Config.BROKER_RULES["lot_step"]
            return SymbolInfo()

    def account_info(self):
        if self.live and MT5_AVAILABLE:
            try:
                return mt5.account_info()
            except Exception:
                return None
        else:
            class AccountInfo:
                def __init__(self) -> None:
                    self.balance = Config.SIM["start_balance"]
                    self.equity = Config.SIM["start_balance"]
                    self.profit = 0.0
            return AccountInfo()

    def order_send(self, request: Dict[str, Any]):
        if self.live and MT5_AVAILABLE:
            try:
                return mt5.order_send(request)
            except Exception as e:
                log(f"خطا در ارسال سفارش: {e}", "error")
                return None
        else:
            class OrderResult:
                def __init__(self, retcode: int, comment: str) -> None:
                    self.retcode = retcode
                    self.comment = comment
                    self.order = random.randint(100000, 999999)
            if random.random() < 0.92:
                return OrderResult(10009, "انجام شد (شبیه‌سازی)")
            else:
                return OrderResult(1, "خطا (شبیه‌سازی)")


mt5i = MT5Adapter(mode=Config.MODE)


def compute_indicators(df: "pd.DataFrame") -> Dict[str, Any]:
    if df is None or df.empty or not NUMPY_PANDAS:
        return {}
    res: Dict[str, Any] = {}
    try:
        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        if TALIB_AVAILABLE:
            res["rsi"] = float(talib.RSI(closes, timeperiod=14)[-1])
            res["atr"] = float(talib.ATR(highs, lows, closes, timeperiod=14)[-1])
            res["ema50"] = float(talib.EMA(closes, timeperiod=50)[-1])
            res["ema200"] = float(talib.EMA(closes, timeperiod=200)[-1])
        else:
            res["rsi"] = 50.0
            if len(closes) > 1:
                tr = max(
                    highs[-1] - lows[-1],
                    max(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2])),
                )
                res["atr"] = float(tr)
            else:
                res["atr"] = 0.01
            res["ema50"] = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(closes[-1])
            res["ema200"] = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(closes[-1])
    except Exception as e:
        log(f"خطا در محاسبه اندیکاتورها: {e}", "error")
    return res


def detect_candle_pattern(df: "pd.DataFrame") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"pattern": None, "strength": "weak"}
    r = df.iloc[-1]
    o, c, h, l = r["open"], r["close"], r["high"], r["low"]
    body = abs(c - o)
    full = h - l + 1e-9
    ratio = body / full
    if ratio > 0.66:
        return {"pattern": "marubozu", "strength": "strong"}
    if ratio > 0.4:
        return {"pattern": "strong_body", "strength": "medium"}
    if ratio < 0.08:
        return {"pattern": "doji", "strength": "weak"}
    return {"pattern": None, "strength": "weak"}


def detect_pinbar(df: "pd.DataFrame") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"is_pinbar": False, "direction": None}
    last_candle = df.iloc[-1]
    o, c, h, l = last_candle["open"], last_candle["close"], last_candle["high"], last_candle["low"]
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range = h - l
    if lower_shadow >= 2 * body and lower_shadow >= 0.6 * total_range:
        return {"is_pinbar": True, "direction": "bullish"}
    elif upper_shadow >= 2 * body and upper_shadow >= 0.6 * total_range:
        return {"is_pinbar": True, "direction": "bearish"}
    return {"is_pinbar": False, "direction": None}


def detect_engulfing(df: "pd.DataFrame") -> Dict[str, Any]:
    if df is None or len(df) < 2:
        return {"is_engulfing": False, "direction": None}
    current = df.iloc[-1]
    prev = df.iloc[-2]
    if current["close"] > current["open"] and prev["close"] < prev["open"]:
        if current["open"] < prev["close"] and current["close"] > prev["open"]:
            return {"is_engulfing": True, "direction": "bullish"}
    elif current["close"] < current["open"] and prev["close"] > prev["open"]:
        if current["open"] > prev["close"] and current["close"] < prev["open"]:
            return {"is_engulfing": True, "direction": "bearish"}
    return {"is_engulfing": False, "direction": None}


def detect_volume_confirmation(df: "pd.DataFrame", ratio: float = 1.5) -> bool:
    if df is None or len(df) < 20:
        return False
    current_volume = float(df.iloc[-1]["tick_volume"])
    avg_volume = float(df["tick_volume"].tail(20).mean())
    return current_volume > avg_volume * ratio


def detect_structure(df: "pd.DataFrame", swing_window: int = 5) -> Dict[str, Any]:
    """تشخیص ساختار بازار با قابلیت شناسایی BOS و CHoCH"""
    if df is None or df.empty or not NUMPY_PANDAS:
        return {"structure": "unknown"}
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)
    swings: List[Tuple[str, int, float]] = []
    for i in range(swing_window, n - swing_window):
        if highs[i] == max(highs[i - swing_window : i + swing_window + 1]):
            swings.append(("high", i, highs[i]))
        if lows[i] == min(lows[i - swing_window : i + swing_window + 1]):
            swings.append(("low", i, lows[i]))
    if not swings:
        return {"structure": "range"}
    if len(swings) >= 4:
        last_swing = swings[-1]
        prev_swing = swings[-2]
        prev2_swing = swings[-3]
        prev3_swing = swings[-4]
        if (
            prev3_swing[0] == "low"
            and prev2_swing[0] == "low"
            and prev_swing[0] == "high"
            and last_swing[0] == "high"
            and prev2_swing[2] < prev3_swing[2]
            and last_swing[2] > prev_swing[2]
        ):
            return {"structure": "CHoCH", "dir": "bull", "swing": last_swing}
        if (
            prev3_swing[0] == "high"
            and prev2_swing[0] == "high"
            and prev_swing[0] == "low"
            and last_swing[0] == "low"
            and prev2_swing[2] > prev3_swing[2]
            and last_swing[2] < prev_swing[2]
        ):
            return {"structure": "CHoCH", "dir": "bear", "swing": last_swing}
    last_swing = swings[-1]
    last_close = float(df["close"].iloc[-1])
    if last_swing[0] == "high" and last_close > float(last_swing[2]):
        return {"structure": "BOS", "dir": "bull", "swing": last_swing}
    if last_swing[0] == "low" and last_close < float(last_swing[2]):
        return {"structure": "BOS", "dir": "bear", "swing": last_swing}
    return {"structure": "range"}


def detect_liquidity_zones(df: "pd.DataFrame", lookback: int = 100) -> Dict[str, Any]:
    if df is None or df.empty or not NUMPY_PANDAS or len(df) < lookback + 2:
        return {"support": [], "resistance": []}
    highs = df["high"].values
    lows = df["low"].values
    support: List[float] = []
    resistance: List[float] = []
    for i in range(lookback, len(df) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            support.append(float(lows[i]))
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistance.append(float(highs[i]))
    support = list(set(round(s, 2) for s in support))
    resistance = list(set(round(r, 2) for r in resistance))
    return {"support": sorted(support), "resistance": sorted(resistance)}


def calc_auto_fib(df: "pd.DataFrame", lookback: int = 200) -> Dict[str, Any]:
    if df is None or df.empty or not NUMPY_PANDAS or len(df) < max(lookback, 2):
        return {
            "fib_0.5": None,
            "fib_0.618": None,
            "fib_0.705": None,
            "golden_zone": (None, None),
            "fib_0.318": None,
        }
    hi = float(df["high"][(-lookback):].max())
    lo = float(df["low"][(-lookback):].min())
    diff = hi - lo if hi >= lo else 0.0
    return {
        "high": hi,
        "low": lo,
        "fib_0.5": lo + 0.5 * diff,
        "fib_0.618": lo + 0.618 * diff,
        "fib_0.705": lo + 0.705 * diff,
        "golden_zone": (lo + Config.GOLDEN_ZONE[0] * diff, lo + Config.GOLDEN_ZONE[1] * diff),
        "fib_0.318": lo + 0.318 * diff,
    }


class CompositeEntryEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {
            "ema": 1.0,
            "rsi": 0.8,
            "candle": 1.5,
            "fib": 1.8,
            "structure": 1.0,
            "volume": 2.0,
            "liquidity": 2.0,
            "pinbar": 1.5,
            "engulfing": 1.5,
            "combo": 2.5,
        }

    def evaluate(self, symbol: str, primary_df: "pd.DataFrame", entry_df: "pd.DataFrame") -> Dict[str, Any]:
        ind_p = compute_indicators(primary_df)
        ind_e = compute_indicators(entry_df)
        candle = detect_candle_pattern(entry_df)
        pinbar = detect_pinbar(entry_df)
        engulfing = detect_engulfing(entry_df)
        volume_ok = detect_volume_confirmation(entry_df)
        struct = detect_structure(primary_df)
        fib = calc_auto_fib(primary_df, lookback=min(200, len(primary_df)))
        liquidity = detect_liquidity_zones(primary_df)
        if entry_df is not None and not entry_df.empty:
            current_price = float(entry_df["close"].iloc[-1])
        else:
            last_data = mt5i.get_rates(symbol, Config.ENTRY_TF, 1)
            current_price = float(last_data["close"].iloc[0]) if last_data is not None and not last_data.empty else 0.0
        if fib.get("golden_zone") is None:
            decision = "no_entry"
            details = {"reason": "Golden zone not available"}
            trade_memory.record_signal(symbol, decision, 0.0, details)
            return {
                "symbol": symbol,
                "decision": decision,
                "score": 0.0,
                "details": details,
                "price": current_price,
                "suggested": {"sl": None, "tp": None},
                "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
            }
        gz_low, gz_high = fib["golden_zone"]
        if gz_low is None or gz_high is None or not (gz_low <= current_price <= gz_high):
            decision = "no_entry"
            details = {"reason": "Price not in golden zone"}
            trade_memory.record_signal(symbol, decision, 0.0, details)
            return {
                "symbol": symbol,
                "decision": decision,
                "score": 0.0,
                "details": details,
                "price": current_price,
                "suggested": {"sl": None, "tp": None},
                "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
            }
        if not volume_ok:
            decision = "no_entry"
            details = {"reason": "Volume confirmation failed"}
            trade_memory.record_signal(symbol, decision, 0.0, details)
            return {
                "symbol": symbol,
                "decision": decision,
                "score": 0.0,
                "details": details,
                "price": current_price,
                "suggested": {"sl": None, "tp": None},
                "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
            }
        if struct.get("structure") not in ["BOS", "CHoCH"]:
            decision = "no_entry"
            details = {"reason": "No BOS/CHoCH structure detected", "structure": struct}
            trade_memory.record_signal(symbol, decision, 0.0, details)
            return {
                "symbol": symbol,
                "decision": decision,
                "score": 0.0,
                "details": details,
                "price": current_price,
                "suggested": {"sl": None, "tp": None},
                "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
            }
        candle_confirmed = False
        candle_details: Dict[str, Any] = {}
        combo_score = 0.0
        if pinbar["is_pinbar"] and engulfing["is_engulfing"]:
            candle_confirmed = True
            candle_details = {"type": "pinbar_engulfing_combo", "direction": pinbar["direction"]}
            combo_score = self.weights["combo"]
        elif pinbar["is_pinbar"] and candle["strength"] == "strong":
            candle_confirmed = True
            candle_details = {"type": "pinbar", "direction": pinbar["direction"]}
            combo_score = self.weights["pinbar"]
        elif engulfing["is_engulfing"]:
            candle_confirmed = True
            candle_details = {"type": "engulfing", "direction": engulfing["direction"]}
            combo_score = self.weights["engulfing"]
        elif candle["strength"] in ["strong", "medium"]:
            last_candle = entry_df.iloc[-1]
            o, c, h, l = last_candle["open"], last_candle["close"], last_candle["high"], last_candle["low"]
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l
            if (upper_shadow >= 0.3 * total_range) or (lower_shadow >= 0.3 * total_range):
                candle_confirmed = True
                candle_details = {"type": "strong_shadow", "direction": "bullish" if c > o else "bearish"}
                combo_score = self.weights["candle"]
        if not candle_confirmed:
            decision = "no_entry"
            details = {"reason": "Candle pattern not strong enough"}
            trade_memory.record_signal(symbol, decision, 0.0, details)
            return {
                "symbol": symbol,
                "decision": decision,
                "score": 0.0,
                "details": details,
                "price": current_price,
                "suggested": {"sl": None, "tp": None},
                "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
            }
        liquidity_support = False
        nearest_support = min(liquidity["support"], key=lambda x: abs(x - current_price)) if liquidity["support"] else None
        nearest_resistance = min(liquidity["resistance"], key=lambda x: abs(x - current_price)) if liquidity["resistance"] else None
        if fib.get("high") is not None and fib.get("low") is not None:
            span = fib["high"] - fib["low"]
            if nearest_support is not None and abs(current_price - nearest_support) < span * 0.05:
                liquidity_support = True
            elif nearest_resistance is not None and abs(current_price - nearest_resistance) < span * 0.05:
                liquidity_support = True
        score = 0.0
        details: Dict[str, Any] = {
            "trend": "unknown",
            "rsi_ok": False,
            "candle": candle_details,
            "in_golden_zone": True,
            "volume_ok": volume_ok,
            "liquidity_support": liquidity_support,
            "structure": struct,
            "pinbar": pinbar,
            "engulfing": engulfing,
        }
        if ind_p.get("ema50") and ind_p.get("ema200"):
            score += self.weights["ema"]
            details["trend"] = "bull" if ind_p["ema50"] > ind_p["ema200"] else "bear"
        rsi = float(ind_e.get("rsi", 50))
        if 30 <= rsi <= 70:
            score += self.weights["rsi"]
            details["rsi_ok"] = True
        score += combo_score
        score += self.weights["fib"]
        score += self.weights["volume"]
        if liquidity_support:
            score += self.weights["liquidity"]
        if struct.get("structure") in ["BOS", "CHoCH"]:
            score += self.weights["structure"]
        max_possible = sum(self.weights.values())
        normalized = score / (max_possible + 1e-9)
        decision = "no_entry"
        if normalized > 0.75:
            decision = "strong_entry"
        elif normalized > 0.45:
            decision = "conditional_entry"
        atr = ind_e.get("atr", 0.0)
        suggested = {"sl": None, "tp": None}
        if atr and details.get("trend") in ["bull", "bear"]:
            if details["trend"] == "bull":
                suggested["sl"] = current_price - atr * 1.5
                suggested["tp"] = current_price + atr * 3.0
            else:
                suggested["sl"] = current_price + atr * 1.5
                suggested["tp"] = current_price - atr * 3.0
        trade_memory.record_signal(symbol, decision, float(normalized), details)
        return {
            "symbol": symbol,
            "decision": decision,
            "score": float(normalized),
            "details": details,
            "price": current_price,
            "suggested": suggested,
            "inds": {"primary": ind_p, "entry": ind_e, "fib": fib},
        }


composite_engine = CompositeEntryEngine()


class RiskManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.prop_mode = False
        self.custom_max_loss = 5.0
        self.daily_loss = 0.0
        self.daily_trades: List[Dict[str, Any]] = []
        self.max_trades = Config.RISK["max_open_trades"]
        self.today = datetime.utcnow().date()

    def set_prop_mode(self, enabled: bool) -> None:
        with self.lock:
            self.prop_mode = enabled
            if enabled:
                Config.RISK["max_daily_loss_pct"] = 2.9
            else:
                Config.RISK["max_daily_loss_pct"] = self.custom_max_loss
            log(f"مدیریت ریسک: حالت Prop {'فعال' if enabled else 'غیرفعال'} شد", "info")

    def calculate_position_size(self, symbol: str, risk_pct: float, entry_price: float, sl_price: float) -> float:
        if not entry_price or not sl_price or entry_price == sl_price:
            return 0.0
        try:
            account = mt5i.account_info()
            balance = getattr(account, "balance", Config.SIM["start_balance"])
            risk_amount = balance * (risk_pct / 100.0)
            point = getattr(mt5i.symbol_info(symbol), "point", 0.0001) or 0.0001
            pip_diff = abs(entry_price - sl_price) / point
            pip_value = 1.0
            lot = risk_amount / max(pip_diff * pip_value, 1e-9)
            lot = max(Config.BROKER_RULES["min_lot"], min(Config.BROKER_RULES["max_lot"], lot))
            lot = round(lot / Config.BROKER_RULES["lot_step"]) * Config.BROKER_RULES["lot_step"]
            return float(lot)
        except Exception as e:
            log(f"خطا در محاسبه حجم معامله: {e}", "error")
            return 0.0

    def check_risk_limits(self, symbol: str, lot: float) -> Tuple[bool, str]:
        with self.lock:
            now = datetime.utcnow()
            if now.date() != self.today:
                self.today = now.date()
                self.daily_loss = 0.0
                self.daily_trades = []
            self.daily_trades = [t for t in self.daily_trades if datetime.fromisoformat(t["ts"]).date() == now.date()]
            self.daily_loss = sum(t.get("pnl", 0.0) for t in self.daily_trades if t.get("pnl", 0.0) < 0)
            account = mt5i.account_info()
            balance = getattr(account, "balance", Config.SIM["start_balance"])
            max_daily_loss = Config.RISK["max_daily_loss_pct"] / 100.0 * balance
            if len(self.daily_trades) >= self.max_trades:
                return False, "تعداد معاملات به حداکثر رسیده است"
            if abs(self.daily_loss) >= max_daily_loss:
                return False, "حداکثر ضرر روزانه رسیده است"
            return True, "OK"

    def record_trade(self, trade: Dict[str, Any]) -> None:
        with self.lock:
            self.daily_trades.append({"ts": datetime.utcnow().isoformat(), "symbol": trade["symbol"], "pnl": trade.get("pnl", 0.0)})


risk_manager = RiskManager()


class OrderManager:
    def __init__(self) -> None:
        self.order_history: List[Dict[str, Any]] = []
        self.trailing_enabled = False
        self.trailing_mult = 1.0
        self.lock = threading.RLock()

    def set_trailing(self, enabled: bool, multiplier: float) -> None:
        with self.lock:
            self.trailing_enabled = enabled
            self.trailing_mult = max(0.5, min(multiplier, 3.0))
            log(f"تریلینگ استاپ {'فعال' if enabled else 'غیرفعال'} شد، ضریب={self.trailing_mult}", "info")

    def execute_order(self, symbol: str, side: str, volume: float, price: float, sl: float, tp: float) -> Dict[str, Any]:
        with self.lock:
            allowed, reason = risk_manager.check_risk_limits(symbol, volume)
            if not allowed:
                log(f"سفارش برای {symbol} رد شد: {reason}", "warning")
                return {"status": "rejected", "reason": reason}
            si = mt5i.symbol_info(symbol)
            if not si:
                log(f"اطلاعات نماد {symbol} یافت نشد", "error")
                return {"status": "error", "reason": "اطلاعات نماد یافت نشد"}
            spread = getattr(si, "spread", 0.0)
            max_spread = Config.BACKTEST["spread_map"].get(symbol, 1.0) * 2
            if spread > max_spread:
                log(f"اسپرد برای {symbol} بسیار بالا است: {spread}", "warning")
                return {"status": "rejected", "reason": "اسپرد بالا"}
            request = {
                "action": mt5.TRADE_ACTION_DEAL if MT5_AVAILABLE else "deal",
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else side,
                "price": price,
                "sl": sl,
                "tp": tp,
                "type_time": mt5.ORDER_TIME_GTC if MT5_AVAILABLE else "gtc",
                "type_filling": mt5.ORDER_FILLING_IOC if MT5_AVAILABLE else "ioc",
            }
            res = mt5i.order_send(request)
            if res and getattr(res, "retcode", 1) == 10009:
                trade = {
                    "symbol": symbol,
                    "side": side,
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "volume": volume,
                    "order_id": getattr(res, "order", random.randint(100000, 999999)),
                    "ts": datetime.utcnow().isoformat(),
                }
                self.order_history.append(trade)
                trade_memory.record_trade(trade)
                risk_manager.record_trade(trade)
                log(f"سفارش اجرا شد: {symbol} {side} حجم={volume} قیمت={price}", "info")
                return {"status": "success", "order_id": trade["order_id"]}
            else:
                error_msg = getattr(res, "comment", "خطای نامشخص") if res else "خطای نامشخص"
                log(f"خطا در اجرای سفارش برای {symbol}: {error_msg}", "error")
                return {"status": "error", "reason": error_msg}

    def close_all(self) -> None:
        with self.lock:
            for trade in self.order_history[:]:
                if trade.get("exit") is None:
                    last_data = mt5i.get_rates(trade["symbol"], Config.ENTRY_TF, 1)
                    if last_data is not None and not last_data.empty:
                        trade["exit"] = float(last_data["close"].iloc[-1])
                        if trade["side"] == "buy":
                            trade["pnl"] = (trade["exit"] - trade["entry"]) * trade["volume"]
                        else:
                            trade["pnl"] = (trade["entry"] - trade["exit"]) * trade["volume"]
                        trade_memory.record_trade(trade)
                        risk_manager.record_trade(trade)
                        self.order_history.remove(trade)
            log("تمام معاملات باز بسته شدند", "info")


order_manager = OrderManager()


class NewsFilter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = Config.CACHE_TTL_SEC

    def is_blocked(self, symbol: str) -> Tuple[bool, str]:
        with self._lock:
            now = datetime.utcnow()
            cached = self._cache.get(symbol)
            if cached and now < cached["ts"] + timedelta(seconds=self._ttl):
                return bool(cached["blocked"]), str(cached["reason"])
            blocked = random.random() < 0.1
            reason = "اخبار مهم" if blocked else "بدون اخبار مهم"
            self._cache[symbol] = {"blocked": blocked, "reason": reason, "ts": now}
            return blocked, reason


news_filter = NewsFilter()


class Watchdog:
    def __init__(self, monitor: "TradingMonitor", interval_sec: int = Config.WATCHDOG_INTERVAL_SEC) -> None:
        self.interval = interval_sec
        self.monitor = monitor
        self.last_beat = time.time()
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def beat(self) -> None:
        self.last_beat = time.time()

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        log("سیستم نظارتی فعال شد", "info")

    def _monitor_loop(self) -> None:
        while self.running:
            current_time = time.time()
            if current_time - self.last_beat > self.interval * 6:
                log("سیستم نظارتی فعال شد: توقف مانیتور", "error")
                try:
                    self.monitor.stop()
                except Exception:
                    pass
                break
            time.sleep(self.interval)

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        log("سیستم نظارتی متوقف شد", "info")


class Dashboard:
    def __init__(self, monitor: "TradingMonitor") -> None:
        if not TKINTER_AVAILABLE:
            log("Tkinter در دسترس نیست. دشبورد غیرفعال شد.", "warning")
            return
        self.monitor = monitor
        self.root = tk.Tk()
        self.root.title("دشبورد ربات تریدینگ حرفه‌ای")
        self.root.geometry("1000x800")
        self._setup_variables()
        self._build_ui()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        threading.Thread(target=self.root.mainloop, daemon=True).start()

    def _setup_variables(self) -> None:
        self.module_states = {m: tk.BooleanVar(value=True) for m in Config.DASHBOARD["modules"]}
        self.prop_mode_var = tk.BooleanVar(value=False)
        self.current_symbol = tk.StringVar(value=Config.SYMBOLS[0])
        self.primary_tf = tk.StringVar(value=Config.PRIMARY_TF)
        self.entry_tf = tk.StringVar(value=Config.ENTRY_TF)
        self.lot_size = tk.DoubleVar(value=0.01)
        self.auto_lot = tk.BooleanVar(value=True)
        self.sl_default = tk.DoubleVar(value=50.0)
        self.tp_default = tk.DoubleVar(value=100.0)
        self.trailing_on = tk.BooleanVar(value=False)
        self.trailing_mult = tk.DoubleVar(value=1.0)
        self.current_session = tk.StringVar(value="Asia")
        self.spread_label = tk.Label(self.root, text="اسپرد: نامعلوم")
        self.atr_label = tk.Label(self.root, text="ATR: نامعلوم")
        self.rsi_label = tk.Label(self.root, text="RSI: نامعلوم")
        self.ema200_label = tk.Label(self.root, text="EMA200: نامعلوم")
        self.price_label = tk.Label(self.root, text="قیمت: نامعلوم")
        self.last_signal = tk.Label(self.root, text="آخرین سیگنال: هیچ")
        self.open_trades = tk.Label(self.root, text="معاملات باز: 0")
        self.pnl_label = tk.Label(self.root, text="سود/زیان: 0.0")
        self.success_pct = tk.Label(self.root, text="درصد موفقیت: نامعلوم")
        self.connection_status = tk.Label(self.root, text="وضعیت اتصال: شبیه‌سازی")
        self.last_update = tk.Label(self.root, text="آخرین به‌روزرسانی: هیچ")
        self.alerts = tk.Label(self.root, text="هشدارها: هیچ")

    def _build_ui(self) -> None:
        market_frame = ttk.LabelFrame(self.root, text="اطلاعات بازار زنده")
        market_frame.pack(padx=10, pady=5, fill="x")
        ttk.Label(market_frame, text="نماد:").grid(row=0, column=0, sticky="w")
        symbol_combo = ttk.Combobox(market_frame, values=Config.SYMBOLS, textvariable=self.current_symbol)
        symbol_combo.grid(row=0, column=1, sticky="ew")
        ttk.Label(market_frame, text="TF تحلیل:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(market_frame, values=["H1", "H4", "M15"], textvariable=self.primary_tf).grid(row=0, column=3, sticky="ew")
        ttk.Label(market_frame, text="TF ورود:").grid(row=0, column=4, sticky="w")
        ttk.Combobox(market_frame, values=["M15", "M5", "M1"], textvariable=self.entry_tf).grid(row=0, column=5, sticky="ew")
        for i, w in enumerate([self.spread_label, self.atr_label, self.rsi_label, self.ema200_label, self.price_label], start=1):
            w.grid(row=i, column=0, columnspan=6, sticky="w")
        settings_frame = ttk.LabelFrame(self.root, text="تنظیمات")
        settings_frame.pack(padx=10, pady=5, fill="x")
        ttk.Checkbutton(settings_frame, text="حجم خودکار (براساس ریسک %)", variable=self.auto_lot).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(settings_frame, text="استاپ‌لاس پیش‌فرض:").grid(row=1, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.sl_default).grid(row=1, column=1, sticky="ew")
        ttk.Label(settings_frame, text="تیک‌پروفیت پیش‌فرض:").grid(row=2, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.tp_default).grid(row=2, column=1, sticky="ew")
        ttk.Checkbutton(settings_frame, text="تریلینگ استاپ", variable=self.trailing_on, command=self._toggle_trailing).grid(row=3, column=0, sticky="w")
        ttk.Label(settings_frame, text="ضریب تریلینگ:").grid(row=3, column=1, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.trailing_mult).grid(row=3, column=2, sticky="ew")
        modules_frame = ttk.LabelFrame(self.root, text="کنترل ماژول‌ها (فعال/غیرفعال)")
        modules_frame.pack(padx=10, pady=5, fill="x")
        for i, module in enumerate(Config.DASHBOARD["modules"]):
            row = i // 3
            col = i % 3
            ttk.Checkbutton(
                modules_frame,
                text=module,
                variable=self.module_states[module],
                command=lambda m=module: self._toggle_module(m),
            ).grid(row=row, column=col, sticky="w", padx=5, pady=2)
        mode_frame = ttk.LabelFrame(self.root, text="انتخاب حالت")
        mode_frame.pack(padx=10, pady=5, fill="x")
        ttk.Checkbutton(
            mode_frame,
            text="حالت Prop (حداکثر ضرر روزانه 2.9%)",
            variable=self.prop_mode_var,
            command=self._toggle_prop_mode,
        ).grid(row=0, column=0, sticky="w")
        self.mode_label = ttk.Label(mode_frame, text="حالت فعلی: واقعی")
        self.mode_label.grid(row=0, column=1, sticky="w")
        signals_frame = ttk.LabelFrame(self.root, text="سیگنال‌ها و وضعیت معاملات")
        signals_frame.pack(padx=10, pady=5, fill="x")
        for w in [self.last_signal, self.open_trades, self.pnl_label, self.success_pct]:
            w.pack(in_=signals_frame, anchor="w", padx=5, pady=2)
        manual_frame = ttk.LabelFrame(self.root, text="مدیریت دستی")
        manual_frame.pack(padx=10, pady=5, fill="x")
        ttk.Button(manual_frame, text="بستن تمام معاملات", command=order_manager.close_all).grid(row=0, column=0, padx=5)
        ttk.Button(manual_frame, text="خرید دستی", command=self._manual_buy).grid(row=0, column=1, padx=5)
        ttk.Button(manual_frame, text="فروش دستی", command=self._manual_sell).grid(row=0, column=2, padx=5)
        tech_frame = ttk.LabelFrame(self.root, text="اطلاعات فنی")
        tech_frame.pack(padx=10, pady=5, fill="x")
        for w in [self.connection_status, self.last_update, self.alerts]:
            w.pack(in_=tech_frame, anchor="w", padx=5, pady=2)

    def _toggle_trailing(self) -> None:
        order_manager.set_trailing(self.trailing_on.get(), self.trailing_mult.get())

    def _toggle_module(self, module: str) -> None:
        state = self.module_states[module].get()
        log(f"وضعیت ماژول {module} تغییر کرد به {'فعال' if state else 'غیرفعال'}", "info")

    def _toggle_prop_mode(self) -> None:
        enabled = self.prop_mode_var.get()
        risk_manager.set_prop_mode(enabled)
        mode = "Prop" if enabled else "واقعی"
        self.mode_label.config(text=f"حالت فعلی: {mode}")
        log(f"حالت به {mode} تغییر کرد", "info")

    def _manual_buy(self) -> None:
        symbol = self.current_symbol.get()
        volume = self.lot_size.get()
        price = self._get_current_price(symbol)
        sl = price - self.sl_default.get()
        tp = price + self.tp_default.get()
        order_manager.execute_order(symbol, "buy", volume, price, sl, tp)

    def _manual_sell(self) -> None:
        symbol = self.current_symbol.get()
        volume = self.lot_size.get()
        price = self._get_current_price(symbol)
        sl = price + self.sl_default.get()
        tp = price - self.tp_default.get()
        order_manager.execute_order(symbol, "sell", volume, price, sl, tp)

    def _get_current_price(self, symbol: str) -> float:
        df = mt5i.get_rates(symbol, Config.ENTRY_TF, 1)
        return float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0

    def _update_loop(self) -> None:
        while True:
            try:
                self._update_dashboard()
                time.sleep(5)
            except Exception as e:
                log(f"خطا در به‌روزرسانی دشبورد: {e}", "error")
                time.sleep(10)

    def _update_dashboard(self) -> None:
        symbol = self.current_symbol.get()
        entry_df = mt5i.get_rates(symbol, self.entry_tf.get(), 100)
        primary_df = mt5i.get_rates(symbol, self.primary_tf.get(), 300)
        if entry_df is not None and not entry_df.empty:
            ind_e = compute_indicators(entry_df)
            ind_p = compute_indicators(primary_df) if primary_df is not None else {}
            self.spread_label.config(text=f"اسپرد: {getattr(mt5i.symbol_info(symbol), 'spread', 'نامعلوم')}")
            self.atr_label.config(text=f"ATR: {ind_e.get('atr', 0.0):.4f}")
            self.rsi_label.config(text=f"RSI: {ind_e.get('rsi', 0.0):.2f}")
            self.ema200_label.config(text=f"EMA200: {ind_p.get('ema200', 0.0):.4f}")
            self.price_label.config(text=f"قیمت: {float(entry_df['close'].iloc[-1]):.4f}")
        account = mt5i.account_info()
        if account:
            pnl = getattr(account, "equity", 0.0) - getattr(account, "balance", 0.0)
            self.pnl_label.config(text=f"سود/زیان: {pnl:.2f}")
        open_count = len([t for t in order_manager.order_history if t.get("exit") is None])
        self.open_trades.config(text=f"معاملات باز: {open_count}")
        signals = trade_memory.query_recent_signals(limit=1)
        if signals:
            sig_id, ts, sym, decision, score = signals[0]
            self.last_signal.config(text=f"آخرین سیگنال: {decision} ({sym}) امتیاز: {float(score):.2f}")
        self.connection_status.config(text=f"وضعیت اتصال: {'زنده' if mt5i.live else 'شبیه‌سازی'}")
        self.last_update.config(text=f"آخرین به‌روزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class TradingMonitor:
    def __init__(self) -> None:
        self.running = False
        self.symbols = Config.SYMBOLS
        self.watchdog = Watchdog(self)
        self.thread: Optional[threading.Thread] = None
        self.dashboard: Optional[Dashboard] = None
        if Tkinter_is_usable():
            self.dashboard = Dashboard(self)

    def start(self) -> None:
        self.running = True
        self.watchdog.start()
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        log("مانیتور معاملاتی شروع شد", "info")

    def stop(self) -> None:
        self.running = False
        self.watchdog.stop()
        order_manager.close_all()
        mt5i.shutdown()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        log("مانیتور معاملاتی متوقف شد", "info")

    def _main_loop(self) -> None:
        one_shot = os.getenv(Config.ONE_SHOT_ENV) is not None
        while self.running:
            try:
                self.watchdog.beat()
                for symbol in self.symbols:
                    blocked, reason = news_filter.is_blocked(symbol)
                    if blocked:
                        log(f"معامله برای {symbol} مسدود شد: {reason}", "warning")
                        continue
                    primary_df = mt5i.get_rates(symbol, Config.PRIMARY_TF, 300)
                    entry_df = mt5i.get_rates(symbol, Config.ENTRY_TF, 100)
                    if primary_df is None or primary_df.empty or entry_df is None or entry_df.empty:
                        log(f"داده‌ای برای {symbol} دریافت نشد", "warning")
                        continue
                    signal = composite_engine.evaluate(symbol, primary_df, entry_df)
                    if signal["decision"] in ["strong_entry", "conditional_entry"]:
                        side = "buy" if signal["details"].get("trend") == "bull" else "sell"
                        price = float(signal["price"])
                        sl = float(signal["suggested"]["sl"]) if signal["suggested"]["sl"] is not None else None
                        tp = float(signal["suggested"]["tp"]) if signal["suggested"]["tp"] is not None else None
                        if sl is None or tp is None:
                            log(f"SL/TP نامعتبر برای {symbol}", "warning")
                            continue
                        lot = risk_manager.calculate_position_size(symbol, Config.RISK["low_risk_pct"], price, sl)
                        if lot <= 0:
                            log(f"حجم معامله نامعتبر برای {symbol}", "warning")
                            continue
                        order_manager.execute_order(symbol, side, lot, price, sl, tp)
                if one_shot:
                    self.stop()
                    break
                time.sleep(Config.LOOP_SLEEP_SEC)
            except Exception as e:
                log(f"خطا در حلقه اصلی: {e}\n{traceback.format_exc()}", "error")
                time.sleep(30)


def Tkinter_is_usable() -> bool:
    return TKINTER_AVAILABLE and Config.USE_DASHBOARD


if __name__ == "__main__":
    monitor = TradingMonitor()
    try:
        monitor.start()
        while monitor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        log("دریافت سیگنال توقف از کاربر", "info")
        monitor.stop()
    except Exception as e:
        log(f"خطای غیرمنتظره: {e}\n{traceback.format_exc()}", "critical")
        monitor.stop()