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

# -------------------------
# Optional libraries - guarded imports
# -------------------------
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
	pd = None  # type: ignore
	np = None  # type: ignore
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

# دشبورد - مدیریت عدم نصب Tkinter
try:
	import tkinter as tk
	from tkinter import ttk, messagebox
	TKINTER_AVAILABLE = True
except ImportError:
	TKINTER_AVAILABLE = False
	print("Tkinter not available. Dashboard disabled.")

# -------------------------
# پیکربندی مرکزی
# -------------------------
class Config:
	MODE = "sim"  # 'sim' یا 'live'
	SYMBOLS = ["XAUUSD", "BTCUSD", "US30", "DJI"]
	
	# نگاشت تایم‌فریم‌های MT5
	TF_MAP = {
		"M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
		"M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
		"M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
		"H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
		"H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240
	}
	
	PRIMARY_TF = "H1"
	ENTRY_TF = "M15"
	CONFIRM_TF = "M5"  # تایم‌فریم تاییدیه
	GOLDEN_ZONE = (0.5, 0.705)  # منطقه طلایی گسترش یافته
	
	# تنظیمات پایه
	DB_FILE = "ultimate_bot_pro.db"
	LOG_FILE = "ultimate_bot_pro.log"
	CACHE_TTL_SEC = 15
	
	# مدیریت ریسک
	RISK = {
		"low_risk_pct": 1.0,
		"high_risk_pct": 2.0,
		"rr_ratio": 2.0,
		"max_daily_loss_pct": 2.9,
		"max_open_trades": 4
	}
	
	# قوانین کارگزاری
	BROKER_RULES = {
		"min_lot": 0.01,
		"max_lot": 100.0,
		"lot_step": 0.01
	}
	
	# تنظیمات بکتست
	BACKTEST = {
		"slippage_pct": 0.0005,
		"spread_map": {"XAUUSD": 0.3, "BTCUSD": 5.0, "US30": 1.0, "DJI": 1.0}
	}
	
	# بهینه‌ساز
	OPTIMIZER = {
		"ga_pop": 20,
		"ga_gens": 20,
		"random_budget": 50,
		"parallel_workers": 4
	}
	
	# یادگیری ماشین
	ML = {
		"retrain_interval_hrs": 24,
		"model_file": "ml_recommender.pkl"
	}
	
	# تنظیمات نظارت
	WATCHDOG_INTERVAL_SEC = 10
	
	# شبیه‌سازی
	SIM = {
		"start_balance": 10000.0,
		"volatility": 0.002
	}
	
	# مدیریت نخ‌ها
	THREAD_POOL = {"max_workers": 4}
	
	# دشبورد
	DASHBOARD = {
		"sessions": ["Asia", "London", "New York"],
		"tf_combos": [
			("H1", "M15"),
			("M15", "M5"),
			("M5", "M1")
		],
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
			"Multi-Symbol Orchestrator"
		]
	}

# -------------------------
# سیستم لاگ‌گیری
# -------------------------

def setup_logger():
	logger = logging.getLogger("ultimate_bot_pro")
	logger.setLevel(logging.INFO)
	
	if not logger.handlers:
		# هندلر فایل
		fh = logging.FileHandler(Config.LOG_FILE)
		fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
		fh.setFormatter(fh_formatter)
		
		# هندلر کنسول
		ch = logging.StreamHandler(sys.stdout)
		ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
		ch.setFormatter(ch_formatter)
		
		logger.addHandler(fh)
		logger.addHandler(ch)
	
	return logger

logger = setup_logger()

def log(msg: str, level: str = "info"):
	"""ثبت پیام در سیستم لاگ‌گیری"""
	getattr(logger, level)(msg)

# -------------------------
# سیستم رویداد (Event Bus)
# -------------------------
class EventBus:
	def __init__(self):
		self._subs: Dict[str, List[Callable[[Any], None]]] = {}
		self._lock = threading.RLock()

	def subscribe(self, topic: str, fn: Callable[[Any], None]):
		"""عضویت در یک موضوع خاص"""
		with self._lock:
			if topic not in self._subs:
				self._subs[topic] = []
			self._subs[topic].append(fn)

	def publish(self, topic: str, payload: Any):
		"""انتشار رویداد برای یک موضوع خاص"""
		handlers = []
		with self._lock:
			handlers = self._subs.get(topic, [])[:]  # ایجاد کپی برای جلوگیری از تغییرات همزمان
		
		for handler in handlers:
			try:
				handler(payload)
			except Exception as e:
				log(f"خطا در پردازش رویداد {topic}: {e}\n{traceback.format_exc()}", "error")

event_bus = EventBus()

# -------------------------
# حافظه معاملات (SQLite)
# -------------------------
class TradeMemory:
	def __init__(self, db_file=Config.DB_FILE):
		self.db = db_file
		self._lock = threading.RLock()
		self._init_db()

	def _init_db(self):
		"""ایجاد جداول پایگاه داده در صورت عدم وجود"""
		with sqlite3.connect(self.db) as conn:
			cur = conn.cursor()
			# جدول معاملات
			cur.execute("""CREATE TABLE IF NOT EXISTS trades (
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
			)""")
			
			# جدول سیگنال‌ها
			cur.execute("""CREATE TABLE IF NOT EXISTS signals (
				id TEXT PRIMARY KEY, 
				ts TEXT, 
				symbol TEXT, 
				decision TEXT, 
				score REAL, 
				details TEXT
			)""")
			
			# جدول ویژگی‌ها
			cur.execute("""CREATE TABLE IF NOT EXISTS features (
				id TEXT PRIMARY KEY, 
				ts TEXT, 
				symbol TEXT, 
				features TEXT, 
				label INTEGER
			)""")
			
			conn.commit()
		log("پایگاه داده حافظه معاملات راه‌اندازی شد", "info")

	def record_trade(self, rec: Dict[str, Any]):
		"""ثبت معامله در پایگاه داده (باز یا بسته)"""
		with self._lock, sqlite3.connect(self.db) as conn:
			cur = conn.cursor()
			trade_id = rec.get("id", str(uuid.uuid4()))
			timestamp = rec.get("ts", datetime.utcnow().isoformat())
			
			cur.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
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
				json.dumps(rec.get("meta", {}))
			))
			conn.commit()
		log(f"معامله ثبت شد: {rec['symbol']} {rec['side']} سود/زیان={rec.get('pnl')}", "info")

	def update_trade_exit(self, trade_id: str, exit_price: float, pnl: float, meta: Optional[Dict[str, Any]] = None):
		"""به‌روزرسانی اطلاعات خروج برای معامله"""
		with self._lock, sqlite3.connect(self.db) as conn:
			cur = conn.cursor()
			cur.execute("UPDATE trades SET exit=?, pnl=?, meta=? WHERE id=?", (
				exit_price,
				pnl,
				json.dumps(meta or {}),
				trade_id
			))
			conn.commit()

	def record_signal(self, symbol: str, decision: str, score: float, details: Dict[str, Any]):
		"""ثبت سیگنال در پایگاه داده"""
		with self._lock, sqlite3.connect(self.db) as conn:
			cur = conn.cursor()
			cur.execute("INSERT INTO signals VALUES (?,?,?,?,?,?)", (
				str(uuid.uuid4()),
				datetime.utcnow().isoformat(),
				symbol,
				decision,
				float(score),
				json.dumps(details)
			))
			conn.commit()

	def query_recent_signals(self, limit=50):
		"""دریافت آخرین سیگنال‌ها"""
		with sqlite3.connect(self.db) as conn:
			cur = conn.cursor()
			cur.execute("SELECT id, ts, symbol, decision, score FROM signals ORDER BY ts DESC LIMIT ?", (limit,))
			return cur.fetchall()

trade_memory = TradeMemory()

# -------------------------
# آداپتور MT5 (زنده یا شبیه‌سازی)
# -------------------------
class MT5Adapter:
	def __init__(self, mode="sim"):
		self.mode = mode
		self.sim: Dict[str, Dict[str, Any]] = {}
		self.live = False
		
		if self.mode == "live" and MT5_AVAILABLE:
			self._init_live()
		else:
			self._init_sim()

	def _init_live(self):
		"""راه‌اندازی اتصال به MT5"""
		try:
			if not mt5.initialize():
				raise ConnectionError("Failed to initialize MT5")
			self.live = True
			log("اتصال به MT5 برقرار شد (حالت زنده)", "info")
		except Exception as e:
			log(f"خطا در اتصال به MT5: {e}. تغییر به حالت شبیه‌سازی", "warning")
			self._init_sim()

	def _init_sim(self):
		"""راه‌اندازی حالت شبیه‌سازی"""
		self.live = False
		for symbol in Config.SYMBOLS:
			self.sim[symbol] = {
				"price": 100.0 + random.random() * 100.0,
				"vol": Config.SIM["volatility"],
				"history": []
			}
		log("حالت شبیه‌سازی فعال شد", "info")

	def shutdown(self):
		"""خاتمه اتصال به MT5"""
		if self.live and MT5_AVAILABLE:
			mt5.shutdown()
			log("اتصال به MT5 قطع شد", "info")

	def get_rates(self, symbol: str, timeframe: str, count: int) -> Optional['pd.DataFrame']:
		"""دریافت داده‌های تاریخی"""
		if not NUMPY_PANDAS:
			return None
		if self.live:
			return self._get_live_rates(symbol, timeframe, count)
		else:
			return self._get_sim_rates(symbol, timeframe, count)

	def _get_live_rates(self, symbol: str, timeframe: str, count: int) -> Optional['pd.DataFrame']:
		"""دریافت داده‌های زنده از MT5"""
		if not MT5_AVAILABLE:
			return None
		tf = Config.TF_MAP.get(timeframe, mt5.TIMEFRAME_M15)
		try:
			bars = mt5.copy_rates_from_pos(symbol, tf, 0, count)
			if bars is None or len(bars) == 0:
				return None
				
			df = pd.DataFrame(bars)
			df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
			return df
		except Exception as e:
			log(f"خطا در دریافت داده‌های زنده: {e}", "error")
			return None

	def _get_sim_rates(self, symbol: str, timeframe: str, count: int) -> 'pd.DataFrame':
		"""تولید داده‌های شبیه‌سازی شده"""
		if symbol not in self.sim:
			self.sim[symbol] = {
				"price": 100.0 + random.random() * 100.0,
				"vol": Config.SIM["volatility"],
				"history": []
			}
			
		s = self.sim[symbol]
		rows: List[Dict[str, Any]] = []
		now = datetime.utcnow()
		
		# تعیین بازه زمانی بر اساس تایم‌فریم
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
			delta = timedelta(minutes=15)  # پیش‌فرض
		
		for i in range(count):
			# تولید داده‌های شبیه‌سازی شده
			p = s["price"] * (1 + random.gauss(0, s["vol"]))
			o = p * (1 + random.uniform(-0.0005, 0.0005))
			c = p * (1 + random.uniform(-0.0005, 0.0005))
			h = max(o, c) * (1 + abs(random.uniform(0, 0.0005)))
			l = min(o, c) * (1 - abs(random.uniform(0, 0.0005)))
			ticks = random.randint(1, 100)
			
			# تعیین تایم‌استمپ
			ts = now - (count - i - 1) * delta
			
			rows.append({
				"time": ts, 
				"open": o, 
				"high": h, 
				"low": l, 
				"close": c, 
				"tick_volume": ticks
			})
			
			# به‌روزرسانی تاریخچه
			s["history"].append({
				"time": ts, 
				"open": o, 
				"high": h, 
				"low": l, 
				"close": c
			})
			
			# تغییر تدریجی قیمت پایه
			s["price"] = s["price"] * (1 + random.gauss(0, s["vol"] * 0.1))
			
		df = pd.DataFrame(rows)
		return df

	def symbol_info(self, symbol: str):
		"""دریافت اطلاعات نماد"""
		if self.live and MT5_AVAILABLE:
			try:
				return mt5.symbol_info(symbol)
			except Exception:
				return None
		else:
			# اطلاعات شبیه‌سازی شده
			class SymbolInfo:
				def __init__(self):
					self.point = 0.01
					self.spread = Config.BACKTEST["spread_map"].get(symbol, 1.0)
					self.volume_min = Config.BROKER_RULES["min_lot"]
					self.volume_max = Config.BROKER_RULES["max_lot"]
					self.volume_step = Config.BROKER_RULES["lot_step"]
					
			return SymbolInfo()

	def account_info(self):
		"""دریافت اطلاعات حساب"""
		if self.live and MT5_AVAILABLE:
			try:
				return mt5.account_info()
			except Exception:
				return None
		else:
			# اطلاعات شبیه‌سازی شده
			class AccountInfo:
				def __init__(self):
					self.balance = Config.SIM["start_balance"]
					self.equity = Config.SIM["start_balance"]
					self.profit = 0.0
					
			return AccountInfo()

	def order_send(self, request: Dict[str, Any]):
		"""ارسال سفارش"""
		if self.live and MT5_AVAILABLE:
			try:
				return mt5.order_send(request)
			except Exception as e:
				log(f"خطا در ارسال سفارش: {e}", "error")
				return None
		else:
			# شبیه‌سازی ارسال سفارش
			class OrderResult:
				def __init__(self, retcode, comment):
					self.retcode = retcode
					self.comment = comment
					self.order = random.randint(100000, 999999)
			
			# شبیه‌سازی موفقیت با احتمال 92%
			if random.random() < 0.92:
				return OrderResult(10009, "انجام شد (شبیه‌سازی)")
			else:
				return OrderResult(1, "خطا (شبیه‌سازی)")

mt5i = MT5Adapter(mode=Config.MODE)

# -------------------------
# اندیکاتورها و تشخیص سیگنال
# -------------------------

def compute_indicators(df: 'pd.DataFrame') -> Dict[str, Any]:
	"""محاسبه اندیکاتورهای تکنیکال"""
	if df is None or df.empty or not NUMPY_PANDAS:
		return {}
	
	res: Dict[str, Any] = {}
	try:
		closes = df['close'].astype(float).values
		highs = df['high'].astype(float).values
		lows = df['low'].astype(float).values
		
		if TALIB_AVAILABLE:
			# محاسبه با TA-Lib
			res['rsi'] = float(talib.RSI(closes, timeperiod=14)[-1])
			res['atr'] = float(talib.ATR(highs, lows, closes, timeperiod=14)[-1])
			res['ema50'] = float(talib.EMA(closes, timeperiod=50)[-1])
			res['ema200'] = float(talib.EMA(closes, timeperiod=200)[-1])
		else:
			# محاسبات ساده در صورت عدم وجود TA-Lib
			res['rsi'] = 50.0  # مقدار پیش‌فرض
			
			# محاسبه ATR تقریبی
			if len(closes) > 1:
				tr = max(highs[-1] - lows[-1], 
						max(abs(highs[-1] - closes[-2]), 
						abs(lows[-1] - closes[-2])))
				res['atr'] = float(tr)
			else:
				res['atr'] = 0.01
				
			# محاسبه EMA تقریبی
			if len(closes) >= 200:
				res['ema200'] = float(np.mean(closes[-200:]))
			else:
				res['ema200'] = float(closes[-1])
			if len(closes) >= 50:
				res['ema50'] = float(np.mean(closes[-50:]))
			else:
				res['ema50'] = float(closes[-1])
				
	except Exception as e:
		log(f"خطا در محاسبه اندیکاتورها: {e}", "error")
		
	return res


def detect_candle_pattern(df: 'pd.DataFrame') -> Dict[str, Any]:
	"""تشخیص الگوهای کندلی"""
	if df is None or df.empty:
		return {"pattern": None, "strength": "weak"}
	
	r = df.iloc[-1]
	o, c, h, l = r['open'], r['close'], r['high'], r['low']
	body = abs(c - o)
	full = h - l + 1e-9  # جلوگیری از تقسیم بر صفر
	ratio = body / full
	
	if ratio > 0.66:
		return {"pattern": "marubozu", "strength": "strong"}
	if ratio > 0.4:
		return {"pattern": "strong_body", "strength": "medium"}
	if ratio < 0.08:
		return {"pattern": "doji", "strength": "weak"}
		
	return {"pattern": None, "strength": "weak"}


def detect_pinbar(df: 'pd.DataFrame') -> Dict[str, Any]:
	"""تشخیص کندل پین‌بار"""
	if df is None or df.empty:
		return {"is_pinbar": False, "direction": None}
	
	last_candle = df.iloc[-1]
	o, c, h, l = last_candle['open'], last_candle['close'], last_candle['high'], last_candle['low']
	body = abs(c - o)
	upper_shadow = h - max(o, c)
	lower_shadow = min(o, c) - l
	total_range = h - l
	
	# پین‌بار صعودی (سایه پایینی بلند)
	if lower_shadow >= 2 * body and lower_shadow >= 0.6 * total_range:
		return {"is_pinbar": True, "direction": "bullish"}
	
	# پین‌بار نزولی (سایه بالایی بلند)
	elif upper_shadow >= 2 * body and upper_shadow >= 0.6 * total_range:
		return {"is_pinbar": True, "direction": "bearish"}
	
	return {"is_pinbar": False, "direction": None}


def detect_engulfing(df: 'pd.DataFrame') -> Dict[str, Any]:
	"""تشخیص کندل اینگالف"""
	if df is None or len(df) < 2:
		return {"is_engulfing": False, "direction": None}
	
	current = df.iloc[-1]
	prev = df.iloc[-2]
	
	# Bullish engulfing
	if current['close'] > current['open'] and prev['close'] < prev['open']:
		if current['open'] < prev['close'] and current['close'] > prev['open']:
			return {"is_engulfing": True, "direction": "bullish"}
	
	# Bearish engulfing
	elif current['close'] < current['open'] and prev['close'] > prev['open']:
		if current['open'] > prev['close'] and current['close'] < prev['open']:
			return {"is_engulfing": True, "direction": "bearish"}
	
	return {"is_engulfing": False, "direction": None}


def detect_volume_confirmation(df: 'pd.DataFrame', ratio=1.5) -> bool:
	"""تاییدیه حجم معاملات"""
	if df is None or len(df) < 2:
		return False
	
	current_volume = df.iloc[-1]['tick_volume']
	avg_volume = df['tick_volume'].tail(20).mean()
	
	return current_volume > avg_volume * ratio


def detect_structure(df: 'pd.DataFrame', swing_window=5) -> Dict[str, Any]:
	"""تشخیص ساختار بازار با قابلیت شناسایی BOS و CHoCH"""
	if df is None or df.empty or not NUMPY_PANDAS:
		return {"structure": "unknown"}
	
	highs = df['high'].values
	lows = df['low'].values
	n = len(highs)
	swings: List[Tuple[str, int, float]] = []
	
	# تشخیص نوسانات
	for i in range(swing_window, n - swing_window):
		if highs[i] == max(highs[i - swing_window:i + swing_window + 1]):
			swings.append(("high", i, float(highs[i])))
		if lows[i] == min(lows[i - swing_window:i + swing_window + 1]):
			swings.append(("low", i, float(lows[i])))
	
	# تشخیص شکست ساختار (BOS) و تغییر روند (CHoCH)
	if not swings:
		return {"structure": "range"}
	
	# تشخیص تغییر روند (CHoCH)
	if len(swings) >= 4:
		last_swing = swings[-1]
		prev_swing = swings[-2]
		prev2_swing = swings[-3]
		prev3_swing = swings[-4]
		
		# Bullish CHoCH: Lower Low -> Higher Low -> Higher High
		if (
			prev3_swing[0] == "low" and prev2_swing[0] == "low" and
			prev_swing[0] == "high" and last_swing[0] == "high" and
			prev2_swing[2] < prev3_swing[2] and
			last_swing[2] > prev_swing[2]
		):
			return {"structure": "CHoCH", "dir": "bull", "swing": last_swing}
		
		# Bearish CHoCH: Higher High -> Lower High -> Lower Low
		if (
			prev3_swing[0] == "high" and prev2_swing[0] == "high" and
			prev_swing[0] == "low" and last_swing[0] == "low" and
			prev2_swing[2] > prev3_swing[2] and
			last_swing[2] < prev_swing[2]
		):
			return {"structure": "CHoCH", "dir": "bear", "swing": last_swing}
	
	# تشخیص شکست ساختار (BOS)
	last_swing = swings[-1]
	last_close = df['close'].iloc[-1]
	
	if last_swing[0] == "high" and last_close > last_swing[2]:
		return {"structure": "BOS", "dir": "bull", "swing": last_swing}
	if last_swing[0] == "low" and last_close < last_swing[2]:
		return {"structure": "BOS", "dir": "bear", "swing": last_swing}
		
	return {"structure": "range"}


def detect_liquidity_zones(df: 'pd.DataFrame', lookback=100) -> Dict[str, Any]:
	"""تشخیص سطوح حمایت و مقاومت (نقدینگی)"""
	if df is None or df.empty or not NUMPY_PANDAS or len(df) < (lookback + 2):
		return {"support": [], "resistance": []}
	
	window = df.tail(lookback + 2)
	highs = window['high'].values
	lows = window['low'].values
	support: List[float] = []
	resistance: List[float] = []
	
	for i in range(1, len(window) - 1):
		if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
			support.append(float(lows[i]))
		if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
			resistance.append(float(highs[i]))
	
	# حذف سطوح تکراری و نزدیک به هم
	support = list(set(round(s, 2) for s in support))
	resistance = list(set(round(r, 2) for r in resistance))
	
	return {
		"support": sorted(support),
		"resistance": sorted(resistance)
	}


def calc_auto_fib(df: 'pd.DataFrame', lookback=200) -> Dict[str, Any]:
	"""محاسبه سطوح فیبوناچی"""
	if df is None or df.empty or not NUMPY_PANDAS or len(df) < lookback:
		return {
			"fib_0.5": None,
			"fib_0.618": None, 
			"fib_0.705": None, 
			"golden_zone": (None, None),
			"fib_0.318": None
		}
	
	window = df.tail(lookback)
	hi = float(window['high'].max())
	lo = float(window['low'].min())
	diff = hi - lo
	
	# محاسبه سطوح فیبوناچی
	return {
		"high": hi,
		"low": lo,
		"fib_0.5": lo + 0.5 * diff,
		"fib_0.618": lo + 0.618 * diff,
		"fib_0.705": lo + 0.705 * diff,
		"golden_zone": (lo + Config.GOLDEN_ZONE[0] * diff, lo + Config.GOLDEN_ZONE[1] * diff),
		"fib_0.318": lo + 0.318 * diff
	}

# -------------------------
# موتور ورود ترکیبی (امتیازدهی)
# -------------------------
class CompositeEntryEngine:
	def __init__(self, weights: Optional[Dict[str, float]] = None):
		# وزن‌های پیش‌فرض برای فاکتورهای مختلف
		self.weights = weights or {
			"ema": 1.0,          # جهت میانگین متحرک
			"rsi": 0.8,          # وضعیت RSI
			"candle": 1.5,       # قدرت کندل
			"fib": 1.8,          # منطقه فیبوناچی
			"structure": 1.0,    # ساختار بازار
			"volume": 2.0,       # حجم معاملات (اهمیت بالا)
			"liquidity": 2.0,    # نقدینگی (اهمیت بالا)
			"pinbar": 1.5,       # پین‌بار
			"engulfing": 1.5,    # اینگالف
			"combo": 2.5,        # ترکیب پینبار و اینگالف (بالاترین اهمیت)
		}

	def evaluate(self, symbol: str, primary_df: 'pd.DataFrame', entry_df: 'pd.DataFrame', confirm_df: 'pd.DataFrame') -> Dict[str, Any]:
		"""ارزیابی شرایط ورود به معامله با شرط اجباری BOS/CHoCH"""
		# محاسبه اندیکاتورها
		ind_p = compute_indicators(primary_df)
		ind_e = compute_indicators(entry_df)
		
		# تشخیص الگوها
		candle = detect_candle_pattern(entry_df)
		pinbar = detect_pinbar(entry_df)
		engulfing = detect_engulfing(entry_df)
		volume_ok = detect_volume_confirmation(confirm_df)
		struct = detect_structure(primary_df)  # تشخیص ساختار در تایم‌فریم تحلیل
		fib = calc_auto_fib(primary_df, lookback=min(200, len(primary_df)))
		liquidity = detect_liquidity_zones(primary_df)
		
		# قیمت فعلی
		if entry_df is not None and not entry_df.empty:
			current_price = float(entry_df['close'].iloc[-1])
		else:
			last_data = mt5i.get_rates(symbol, Config.ENTRY_TF, 1)
			if last_data is not None and not last_data.empty:
				current_price = float(last_data['close'].iloc[0])
			else:
				current_price = 0.0
		
		# 1. شرط اجباری: حضور در گلدن زون
		if fib.get('golden_zone', (None, None))[0] is None or fib.get('golden_zone', (None, None))[1] is None:
			decision = 'no_entry'
			details = {"reason": "Golden zone not available"}
			trade_memory.record_signal(symbol, decision, 0.0, details)
			return {
				"symbol": symbol,
				"decision": decision,
				"score": 0.0,
				"details": details,
				"price": current_price,
				"suggested": {"sl": None, "tp": None},
				"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
			}
		
		if not (fib['golden_zone'][0] <= current_price <= fib['golden_zone'][1]):
			decision = 'no_entry'
			details = {"reason": "Price not in golden zone"}
			trade_memory.record_signal(symbol, decision, 0.0, details)
			return {
				"symbol": symbol,
				"decision": decision,
				"score": 0.0,
				"details": details,
				"price": current_price,
				"suggested": {"sl": None, "tp": None},
				"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
			}
		
		# 2. شرط اجباری: تاییدیه حجم
		if not volume_ok:
			decision = 'no_entry'
			details = {"reason": "Volume confirmation failed"}
			trade_memory.record_signal(symbol, decision, 0.0, details)
			return {
				"symbol": symbol,
				"decision": decision,
				"score": 0.0,
				"details": details,
				"price": current_price,
				"suggested": {"sl": None, "tp": None},
				"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
			}
		
		# 3. شرط اجباری: تشخیص BOS یا CHoCH در تایم‌فریم تحلیل
		if struct.get('structure') not in ['BOS', 'CHoCH']:
			decision = 'no_entry'
			details = {"reason": "No BOS/CHoCH structure detected", "structure": struct}
			trade_memory.record_signal(symbol, decision, 0.0, details)
			return {
				"symbol": symbol,
				"decision": decision,
				"score": 0.0,
				"details": details,
				"price": current_price,
				"suggested": {"sl": None, "tp": None},
				"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
			}
		
		# 4. تاییدیه ترکیبی پینبار و اینگالف
		candle_confirmed = False
		candle_details: Dict[str, Any] = {}
		combo_score = 0.0
		
		# گزینه 1: ترکیب پینبار و اینگالف (امتیاز بالا)
		if pinbar["is_pinbar"] and engulfing["is_engulfing"]:
			candle_confirmed = True
			candle_details = {"type": "pinbar_engulfing_combo", "direction": pinbar["direction"]}
			combo_score = self.weights["combo"]
		
		# گزینه 2: پینبار معتبر
		elif pinbar["is_pinbar"] and candle["strength"] == "strong":
			candle_confirmed = True
			candle_details = {"type": "pinbar", "direction": pinbar["direction"]}
			combo_score = self.weights["pinbar"]
		
		# گزینه 3: کندل اینگالف معتبر
		elif engulfing["is_engulfing"]:
			candle_confirmed = True
			candle_details = {"type": "engulfing", "direction": engulfing["direction"]}
			combo_score = self.weights["engulfing"]
		
		# گزینه 4: کندل قوی با شدو معتبر
		elif candle["strength"] in ["strong", "medium"]:
			last_candle = entry_df.iloc[-1]
			o, c, h, l = last_candle['open'], last_candle['close'], last_candle['high'], last_candle['low']
			upper_shadow = h - max(o, c)
			lower_shadow = min(o, c) - l
			total_range = h - l
			
			if (upper_shadow >= 0.3 * total_range) or (lower_shadow >= 0.3 * total_range):
				candle_confirmed = True
				candle_details = {"type": "strong_shadow", "direction": "bullish" if c > o else "bearish"}
				combo_score = self.weights["candle"]
		
		if not candle_confirmed:
			decision = 'no_entry'
			details = {"reason": "Candle pattern not strong enough"}
			trade_memory.record_signal(symbol, decision, 0.0, details)
			return {
				"symbol": symbol,
				"decision": decision,
				"score": 0.0,
				"details": details,
				"price": current_price,
				"suggested": {"sl": None, "tp": None},
				"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
			}
		
		# 5. امتیازدهی برای نقدینگی و حمایت/مقاومت
		liquidity_support = False
		nearest_support = min(liquidity["support"], key=lambda x: abs(x - current_price)) if liquidity["support"] else None
		nearest_resistance = min(liquidity["resistance"], key=lambda x: abs(x - current_price)) if liquidity["resistance"] else None
		
		price_range = (fib.get('high', current_price) - fib.get('low', current_price))
		threshold = price_range * 0.05 if price_range else 0
		
		if nearest_support is not None and threshold and abs(current_price - nearest_support) < threshold:
			liquidity_support = True
		elif nearest_resistance is not None and threshold and abs(current_price - nearest_resistance) < threshold:
			liquidity_support = True
		
		# محاسبه امتیاز (فقط اگر در گلدن زون و کندل قوی باشد)
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
			"engulfing": engulfing
		}
		
		# امتیازدهی بر اساس جهت روند
		if ind_p.get('ema50') and ind_p.get('ema200'):
			if ind_p['ema50'] > ind_p['ema200']:
				score += self.weights['ema']
				details['trend'] = 'bull'
			else:
				score += self.weights['ema']
				details['trend'] = 'bear'
		
		# امتیازدهی بر اساس RSI
		rsi = ind_e.get('rsi', 50)
		if 30 <= rsi <= 70:
			score += self.weights['rsi']
			details['rsi_ok'] = True
		
		# امتیازدهی بر اساس الگوهای کندلی
		score += combo_score
		
		# امتیازدهی بر اساس منطقه فیبوناچی
		score += self.weights['fib']
		
		# امتیازدهی بر اساس حجم
		score += self.weights['volume']
		
		# امتیازدهی بر اساس نقدینگی
		if liquidity_support:
			score += self.weights['liquidity']
		
		# امتیازدهی بر اساس ساختار بازار
		if struct.get('structure') in ['BOS', 'CHoCH']:
			score += self.weights['structure']
		
		# نرمال‌سازی امتیاز
		max_possible = sum(self.weights.values())
		normalized = score / (max_possible + 1e-9)
		
		# تصمیم‌گیری نهایی
		decision = 'no_entry'
		if normalized > 0.75: 
			decision = 'strong_entry'
		elif normalized > 0.45: 
			decision = 'conditional_entry'
		
		# محاسبه SL و TP پیشنهادی
		atr = ind_e.get('atr', 0.0)
		suggested = {'sl': None, 'tp': None}
		
		if atr and details.get('trend') in ['bull','bear']:
			if details['trend'] == 'bull':
				suggested['sl'] = current_price - atr * 1.5
				suggested['tp'] = current_price + atr * 3.0
			else:
				suggested['sl'] = current_price + atr * 1.5
				suggested['tp'] = current_price - atr * 3.0
		
		# ثبت سیگنال
		trade_memory.record_signal(symbol, decision, float(normalized), details)
		
		return {
			"symbol": symbol,
			"decision": decision,
			"score": float(normalized),
			"details": details,
			"price": current_price,
			"suggested": suggested,
			"inds": {"primary": ind_p, "entry": ind_e, "fib": fib}
		}

composite_engine = CompositeEntryEngine()

# -------------------------
# مدیریت ریسک (پیشرفته)
# -------------------------
class RiskManager:
	def __init__(self):
		self.lock = threading.RLock()
		self.prop_mode = False
		self.custom_max_loss = 5.0
		self.daily_loss = 0.0
		self.daily_trades: List[Dict[str, Any]] = []  # only closed trades
		self.max_trades = Config.RISK["max_open_trades"]
		self.today = datetime.utcnow().date()

	def set_prop_mode(self, enabled: bool):
		"""تنظیم حالت Prop"""
		with self.lock:
			self.prop_mode = enabled
			if enabled:
				Config.RISK['max_daily_loss_pct'] = 2.9
			else:
				Config.RISK['max_daily_loss_pct'] = self.custom_max_loss
			log(f"مدیریت ریسک: حالت Prop {'فعال' if enabled else 'غیرفعال'} شد", "info")

	def calculate_position_size(self, symbol: str, risk_pct: float, entry_price: float, sl_price: float) -> float:
		"""محاسبه حجم معامله"""
		if not entry_price or not sl_price or entry_price == sl_price:
			return 0.0
			
		try:
			account = mt5i.account_info()
			balance = getattr(account, 'balance', Config.SIM["start_balance"])
			
			# محاسبه مقدار ریسک
			risk_amount = balance * (risk_pct / 100.0)
			
			# محاسبه فاصله تا استاپ‌لاس
			point = getattr(mt5i.symbol_info(symbol), 'point', 0.0001) or 0.0001
			pip_diff = abs(entry_price - sl_price) / point
			pip_value = 1.0  # مقدار ثابت برای سادگی
			
			# محاسبه حجم
			lot = risk_amount / max(pip_diff * pip_value, 1e-9)
			
			# اعمال محدودیت‌های کارگزاری
			lot = max(Config.BROKER_RULES["min_lot"], 
					 min(Config.BROKER_RULES["max_lot"], lot))
			
			# گرد کردن به نزدیک‌ترین گام
			lot = round(lot / Config.BROKER_RULES["lot_step"]) * Config.BROKER_RULES["lot_step"]
			
			return float(lot)
		except Exception as e:
			log(f"خطا در محاسبه حجم معامله: {e}", "error")
			return 0.0

	def _reset_if_new_day(self):
		"""ریست روزانه"""
		now = datetime.utcnow().date()
		if now != self.today:
			self.today = now
			self.daily_loss = 0.0
			self.daily_trades = []

	def check_risk_limits(self, symbol: str, lot: float) -> Tuple[bool, str]:
		"""بررسی محدودیت‌های ریسک"""
		with self.lock:
			self._reset_if_new_day()
			
			# محاسبه ضرر روزانه از معاملات بسته شده امروز
			self.daily_loss = sum(t['pnl'] for t in self.daily_trades if t.get('pnl', 0) < 0)
			
			account = mt5i.account_info()
			balance = getattr(account, 'balance', Config.SIM["start_balance"])
			max_daily_loss = Config.RISK["max_daily_loss_pct"] / 100.0 * balance
			
			# محدودیت تعداد معاملات باز همزمان
			try:
				open_count = len([t for t in order_manager.order_history if t.get("exit") is None])  # type: ignore[name-defined]
			except Exception:
				open_count = 0
			
			if open_count >= self.max_trades:
				return False, "تعداد معاملات باز همزمان به حداکثر رسیده است"
			if abs(self.daily_loss) >= max_daily_loss:
				return False, "حداکثر ضرر روزانه رسیده است"
				
			return True, "OK"

	def record_trade(self, trade: Dict[str, Any]):
		"""ثبت معامله بسته‌شده در تاریخچه روزانه"""
		with self.lock:
			# تنها معاملات بسته‌شده را لحاظ کن
			if trade.get("exit") is None or trade.get("pnl") is None:
				return
			self._reset_if_new_day()
			self.daily_trades.append({
				"ts": datetime.utcnow().isoformat(),
				"symbol": trade["symbol"],
				"pnl": float(trade.get("pnl", 0.0))
			})

risk_manager = RiskManager()

# -------------------------
# مدیریت سفارشات
# -------------------------
class OrderManager:
	def __init__(self):
		self.order_history: List[Dict[str, Any]] = []
		self.trailing_enabled = False
		self.trailing_mult = 1.0
		self.lock = threading.RLock()

	def set_trailing(self, enabled: bool, multiplier: float):
		"""تنظیم تریلینگ استاپ"""
		with self.lock:
			self.trailing_enabled = enabled
			self.trailing_mult = max(0.5, min(multiplier, 3.0))
			log(f"تریلینگ استاپ {'فعال' if enabled else 'غیرفعال'} شد، ضریب={self.trailing_mult}", "info")

	def execute_order(self, symbol: str, side: str, volume: float, price: float, sl: float, tp: float) -> Dict[str, Any]:
		"""اجرای سفارش"""
		with self.lock:
			# بررسی محدودیت‌های ریسک
			allowed, reason = risk_manager.check_risk_limits(symbol, volume)
			if not allowed:
				log(f"سفارش برای {symbol} رد شد: {reason}", "warning")
				return {"status": "rejected", "reason": reason}
			
			# دریافت اطلاعات نماد
			si = mt5i.symbol_info(symbol)
			if not si:
				log(f"اطلاعات نماد {symbol} یافت نشد", "error")
				return {"status": "error", "reason": "اطلاعات نماد یافت نشد"}
			
			# بررسی اسپرد
			spread = getattr(si, 'spread', 0)
			max_spread = Config.BACKTEST["spread_map"].get(symbol, 1.0) * 2
			if spread > max_spread:
				log(f"اسپرد برای {symbol} بسیار بالا است: {spread}", "warning")
				return {"status": "rejected", "reason": "اسپرد بالا"}
			
			# آماده‌سازی درخواست سفارش
			request = {
				"action": mt5.TRADE_ACTION_DEAL if MT5_AVAILABLE else "deal",
				"symbol": symbol,
				"volume": volume,
				"type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else ("buy" if side == "buy" else "sell"),
				"price": price,
				"sl": sl,
				"tp": tp,
				"type_time": mt5.ORDER_TIME_GTC if MT5_AVAILABLE else "gtc",
				"type_filling": mt5.ORDER_FILLING_IOC if MT5_AVAILABLE else "ioc"
			}
			
			# ارسال سفارش
			res = mt5i.order_send(request)
			if res and getattr(res, 'retcode', 1) == 10009:
				trade_id = str(uuid.uuid4())
				trade = {
					"id": trade_id,
					"symbol": symbol,
					"side": side,
					"entry": price,
					"sl": sl,
					"tp": tp,
					"volume": volume,
					"order_id": getattr(res, 'order', random.randint(100000, 999999)),
					"ts": datetime.utcnow().isoformat()
				}
				self.order_history.append(trade)
				trade_memory.record_trade(trade)
				log(f"سفارش اجرا شد: {symbol} {side} حجم={volume} قیمت={price}", "info")
				return {"status": "success", "order_id": trade["order_id"], "trade_id": trade_id}
			else:
				error_msg = getattr(res, 'comment', 'خطای نامشخص') if res else 'خطای نامشخص'
				log(f"خطا در اجرای سفارش برای {symbol}: {error_msg}", "error")
				return {"status": "error", "reason": error_msg}

	def close_all(self):
		"""بستن تمام معاملات باز"""
		with self.lock:
			for trade in self.order_history[:]:
				if trade.get("exit") is None:
					last_data = mt5i.get_rates(trade["symbol"], Config.ENTRY_TF, 1)
					if last_data is not None and not last_data.empty:
						exit_price = float(last_data['close'].iloc[-1])
						trade["exit"] = exit_price
						if trade["side"] == "buy":
							trade["pnl"] = (exit_price - trade["entry"]) * trade["volume"]
						else:
							trade["pnl"] = (trade["entry"] - exit_price) * trade["volume"]
						trade_memory.update_trade_exit(trade["id"], trade["exit"], trade["pnl"])  # type: ignore[index]
						risk_manager.record_trade(trade)
						self.order_history.remove(trade)
			
			log("تمام معاملات باز بسته شدند", "info")

order_manager = OrderManager()

# -------------------------
# فیلتر اخبار
# -------------------------
class NewsFilter:
	def __init__(self):
		self._lock = threading.RLock()
		self._cache: Dict[str, Dict[str, Any]] = {}
		self._ttl = Config.CACHE_TTL_SEC

	def is_blocked(self, symbol: str) -> Tuple[bool, str]:
		"""بررسی وجود اخبار مهم"""
		with self._lock:
			now = datetime.utcnow()
			cached = self._cache.get(symbol)
			
			if cached and now < cached["ts"] + timedelta(seconds=self._ttl):
				return cached["blocked"], cached["reason"]
			
			# شبیه‌سازی - 10% احتمال وجود اخبار مهم
			blocked = random.random() < 0.1
			reason = "اخبار مهم" if blocked else "بدون اخبار مهم"
			
			self._cache[symbol] = {
				"blocked": blocked,
				"reason": reason,
				"ts": now
			}
			
			return blocked, reason

news_filter = NewsFilter()

# -------------------------
# سیستم نظارتی و ایمنی (Watchdog)
# -------------------------
class Watchdog:
	def __init__(self, monitor, interval_sec=Config.WATCHDOG_INTERVAL_SEC):
		self.interval = interval_sec
		self.monitor = monitor
		self.last_beat = time.time()
		self.thread: Optional[threading.Thread] = None
		self.running = False

	def beat(self):
		"""به‌روزرسانی زمان آخرین فعالیت"""
		self.last_beat = time.time()

	def start(self):
		"""شروع نظارت"""
		self.running = True
		self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
		self.thread.start()
		log("سیستم نظارتی فعال شد", "info")

	def _monitor_loop(self):
		"""حلقه نظارتی"""
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

	def stop(self):
		"""توقف نظارت"""
		self.running = False
		if self.thread and self.thread.is_alive():
			self.thread.join(timeout=2.0)
		log("سیستم نظارتی متوقف شد", "info")

# -------------------------
# دشبورد (Tkinter) - اختیاری
# -------------------------
class Dashboard:
	def __init__(self, monitor):
		if not TKINTER_AVAILABLE:
			log("Tkinter در دسترس نیست. دشبورد غیرفعال شد.", "warning")
			return
			
		self.monitor = monitor
		self.root = tk.Tk()
		self.root.title("دشبورد ربات تریدینگ حرفه‌ای")
		self.root.geometry("1000x800")
		
		# متغیرهای رابط کاربری
		self._setup_variables()
		
		# ساخت رابط کاربری
		self._build_ui()
		
		# شروع به‌روزرسانی خودکار
		self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
		self.update_thread.start()
		
		# شروع حلقه اصلی
		self.root.mainloop()

	def _setup_variables(self):
		"""تنظیم متغیرهای رابط کاربری"""
		self.module_states = {m: tk.BooleanVar(value=True) for m in Config.DASHBOARD["modules"]}
		self.prop_mode_var = tk.BooleanVar(value=False)
		self.current_symbol = tk.StringVar(value=Config.SYMBOLS[0])
		self.primary_tf = tk.StringVar(value=Config.PRIMARY_TF)
		self.entry_tf = tk.StringVar(value=Config.ENTRY_TF)
		self.confirm_tf = tk.StringVar(value=Config.CONFIRM_TF)
		self.lot_size = tk.DoubleVar(value=0.01)
		self.auto_lot = tk.BooleanVar(value=True)
		self.sl_default = tk.DoubleVar(value=50.0)
		self.tp_default = tk.DoubleVar(value=100.0)
		self.trailing_on = tk.BooleanVar(value=False)
		self.trailing_mult = tk.DoubleVar(value=1.0)
		self.current_session = tk.StringVar(value="Asia")
		
		# برچسب‌های وضعیت
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

	def _build_ui(self):
		"""ساخت رابط کاربری"""
		# 1. اطلاعات بازار زنده
		market_frame = ttk.LabelFrame(self.root, text="اطلاعات بازار زنده")
		market_frame.pack(padx=10, pady=5, fill="x")
		
		# نماد و تایم‌فریم‌ها
		ttk.Label(market_frame, text="نماد:").grid(row=0, column=0, sticky="w")
		self.symbol_entry = ttk.Entry(market_frame)
		self.symbol_entry.insert(0, ",".join(Config.SYMBOLS))
		self.symbol_entry.grid(row=0, column=1, sticky="ew")
		ttk.Button(market_frame, text="بروزرسانی نمادها", command=self._update_symbols).grid(row=0, column=2, padx=5)
		
		market_frame.columnconfigure(1, weight=1)
		
		# 2. تنظیمات معاملات
		settings_frame = ttk.LabelFrame(self.root, text="تنظیمات معاملات")
		settings_frame.pack(padx=10, pady=5, fill="x")
		
		ttk.Label(settings_frame, text="حجم ثابت:").grid(row=0, column=0, sticky="w")
		ttk.Entry(settings_frame, textvariable=self.lot_size).grid(row=0, column=1, sticky="ew")
		ttk.Checkbutton(settings_frame, text="حجم خودکار (براساس ریسک %)", variable=self.auto_lot).grid(row=0, column=2, sticky="w")
		
		ttk.Label(settings_frame, text="استاپ‌لاس پیش‌فرض:").grid(row=1, column=0, sticky="w")
		ttk.Entry(settings_frame, textvariable=self.sl_default).grid(row=1, column=1, sticky="ew")
		
		ttk.Label(settings_frame, text="تیک‌پروفیت پیش‌فرض:").grid(row=2, column=0, sticky="w")
		ttk.Entry(settings_frame, textvariable=self.tp_default).grid(row=2, column=1, sticky="ew")
		
		ttk.Checkbutton(settings_frame, text="تریلینگ استاپ", variable=self.trailing_on, command=self._toggle_trailing).grid(row=3, column=0, sticky="w")
		ttk.Label(settings_frame, text="ضریب تریلینگ:").grid(row=3, column=1, sticky="w")
		ttk.Entry(settings_frame, textvariable=self.trailing_mult).grid(row=3, column=2, sticky="ew")
		
		settings_frame.columnconfigure(1, weight=1)
		
		# 3. کنترل ماژول‌ها
		modules_frame = ttk.LabelFrame(self.root, text="کنترل ماژول‌ها (فعال/غیرفعال)")
		modules_frame.pack(padx=10, pady=5, fill="x")
		
		for i, module in enumerate(Config.DASHBOARD["modules"]):
			row = i // 3
			col = i % 3
			ttk.Checkbutton(modules_frame, text=module, variable=self.module_states[module],
						command=lambda m=module: self._toggle_module(m)).grid(row=row, column=col, sticky="w", padx=5, pady=2)
		
		# 4. انتخاب حالت
		mode_frame = ttk.LabelFrame(self.root, text="انتخاب حالت")
		mode_frame.pack(padx=10, pady=5, fill="x")
		
		ttk.Checkbutton(mode_frame, text="حالت Prop (حداکثر ضرر روزانه 2.9%)", 
					variable=self.prop_mode_var, command=self._toggle_prop_mode).grid(row=0, column=0, sticky="w")
		self.mode_label = ttk.Label(mode_frame, text="حالت فعلی: واقعی")
		self.mode_label.grid(row=0, column=1, sticky="w")
		
		# 5. سیگنال‌ها و وضعیت معاملات
		signals_frame = ttk.LabelFrame(self.root, text="سیگنال‌ها و وضعیت معاملات")
		signals_frame.pack(padx=10, pady=5, fill="x")
		
		self.last_signal.pack(in_=signals_frame, anchor="w", padx=5, pady=2)
		self.open_trades.pack(in_=signals_frame, anchor="w", padx=5, pady=2)
		self.pnl_label.pack(in_=signals_frame, anchor="w", padx=5, pady=2)
		self.success_pct.pack(in_=signals_frame, anchor="w", padx=5, pady=2)
		
		# 6. مدیریت دستی
		manual_frame = ttk.LabelFrame(self.root, text="مدیریت دستی")
		manual_frame.pack(padx=10, pady=5, fill="x")
		
		ttk.Button(manual_frame, text="بستن تمام معاملات", command=order_manager.close_all).grid(row=0, column=0, padx=5)
		ttk.Button(manual_frame, text="خرید دستی", command=self._manual_buy).grid(row=0, column=1, padx=5)
		ttk.Button(manual_frame, text="فروش دستی", command=self._manual_sell).grid(row=0, column=2, padx=5)
		
		# 7. اطلاعات فنی
		tech_frame = ttk.LabelFrame(self.root, text="اطلاعات فنی")
		tech_frame.pack(padx=10, pady=5, fill="x")
		
		self.connection_status.pack(in_=tech_frame, anchor="w", padx=5, pady=2)
		self.last_update.pack(in_=tech_frame, anchor="w", padx=5, pady=2)
		self.alerts.pack(in_=tech_frame, anchor="w", padx=5, pady=2)

	def _update_symbols(self):
		"""بروزرسانی نمادها"""
		symbols = self.symbol_entry.get().split(",")
		symbols = [s.strip() for s in symbols if s.strip()]
		
		if len(symbols) > 4:
			messagebox.showwarning("هشدار", "حداکثر 4 نماد مجاز است")
			return
			
		Config.SYMBOLS = symbols
		self.monitor.symbols = symbols
		log(f"نمادها بروزرسانی شدند: {Config.SYMBOLS}", "info")

	def _toggle_trailing(self):
		"""تغییر وضعیت تریلینگ استاپ"""
		order_manager.set_trailing(self.trailing_on.get(), self.trailing_mult.get())

	def _toggle_module(self, module):
		"""تغییر وضعیت ماژول"""
		state = self.module_states[module].get()
		log(f"وضعیت ماژول {module} تغییر کرد به {'فعال' if state else 'غیرفعال'}", "info")

	def _toggle_prop_mode(self):
		"""تغییر حالت Prop"""
		enabled = self.prop_mode_var.get()
		risk_manager.set_prop_mode(enabled)
		mode = "Prop" if enabled else "واقعی"
		self.mode_label.config(text=f"حالت فعلی: {mode}")
		log(f"حالت به {mode} تغییر کرد", "info")

	def _manual_buy(self):
		"""خرید دستی"""
		symbol = self.current_symbol.get()
		volume = self.lot_size.get()
		price = self._get_current_price(symbol)
		sl = price - self.sl_default.get()
		tp = price + self.tp_default.get()
		
		order_manager.execute_order(symbol, "buy", volume, price, sl, tp)

	def _manual_sell(self):
		"""فروش دستی"""
		symbol = self.current_symbol.get()
		volume = self.lot_size.get()
		price = self._get_current_price(symbol)
		sl = price + self.sl_default.get()
		tp = price - self.tp_default.get()
		
		order_manager.execute_order(symbol, "sell", volume, price, sl, tp)

	def _get_current_price(self, symbol):
		"""دریافت قیمت فعلی"""
		df = mt5i.get_rates(symbol, Config.ENTRY_TF, 1)
		return float(df['close'].iloc[-1]) if df is not None and not df.empty else 0.0

	def _update_loop(self):
		"""حلقه به‌روزرسانی خودکار دشبورد"""
		while True:
			try:
				self._update_dashboard()
				time.sleep(5)
			except Exception as e:
				log(f"خطا در به‌روزرسانی دشبورد: {e}", "error")
				time.sleep(10)

	def _update_dashboard(self):
		"""به‌روزرسانی اطلاعات دشبورد"""
		symbol = self.current_symbol.get()
		
		# دریافت داده‌ها
		entry_df = mt5i.get_rates(symbol, self.entry_tf.get(), 100)
		primary_df = mt5i.get_rates(symbol, self.primary_tf.get(), 300)
		
		# به‌روزرسانی اندیکاتورها
		if entry_df is not None and not entry_df.empty:
			ind_e = compute_indicators(entry_df)
			ind_p = compute_indicators(primary_df) if primary_df is not None and not primary_df.empty else {}
			
			si = mt5i.symbol_info(symbol)
			spread_text = getattr(si, 'spread', 'نامعلوم')
			self.spread_label.config(text=f"اسپرد: {spread_text}")
			self.atr_label.config(text=f"ATR: {ind_e.get('atr', 0.0):.4f}")
			self.rsi_label.config(text=f"RSI: {ind_e.get('rsi', 0.0):.2f}")
			self.ema200_label.config(text=f"EMA200: {ind_p.get('ema200', 0.0):.4f}" if ind_p else "EMA200: نامعلوم")
			self.price_label.config(text=f"قیمت: {entry_df['close'].iloc[-1]:.4f}")
		
		# به‌روزرسانی حساب
		account = mt5i.account_info()
		if account:
			pnl = getattr(account, 'equity', 0) - getattr(account, 'balance', 0)
			self.pnl_label.config(text=f"سود/زیان: {pnl:.2f}")
		
		# به‌روزرسانی معاملات باز
		open_count = len([t for t in order_manager.order_history if t.get("exit") is None])
		self.open_trades.config(text=f"معاملات باز: {open_count}")
		
		# به‌روزرسانی آخرین سیگنال
		signals = trade_memory.query_recent_signals(limit=1)
		if signals:
			sig_id, ts, sym, decision, score = signals[0]
			self.last_signal.config(text=f"آخرین سیگنال: {decision} ({sym}) امتیاز: {score:.2f}")
		
		# به‌روزرسانی وضعیت اتصال
		self.connection_status.config(text=f"وضعیت اتصال: {'زنده' if mt5i.live else 'شبیه‌سازی'}")
		
		# به‌روزرسانی زمان آخرین به‌روزرسانی
		self.last_update.config(text=f"آخرین به‌روزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------------
# مانیتور اصلی
# -------------------------
class TradingMonitor:
	def __init__(self):
		self.running = False
		self.symbols = Config.SYMBOLS
		self.watchdog = Watchdog(self)
		self.thread: Optional[threading.Thread] = None
		self.dashboard = None
		
		if TKINTER_AVAILABLE:
			self.dashboard = Dashboard(self)

	def start(self):
		"""شروع مانیتورینگ"""
		self.running = True
		self.watchdog.start()
		self.thread = threading.Thread(target=self._main_loop, daemon=True)
		self.thread.start()
		log("مانیتور معاملاتی شروع شد", "info")

	def stop(self):
		"""توقف مانیتورینگ"""
		self.running = False
		self.watchdog.stop()
		order_manager.close_all()
		mt5i.shutdown()
		
		if self.thread and self.thread.is_alive():
			self.thread.join(timeout=5.0)
			
		log("مانیتور معاملاتی متوقف شد", "info")

	def _process_symbol(self, symbol: str):
		"""پردازش یک نماد - ارزیابی و اجرای معامله در صورت نیاز"""
		# بررسی فیلتر اخبار
		blocked, reason = news_filter.is_blocked(symbol)
		if blocked:
			log(f"معامله برای {symbol} مسدود شد: {reason}", "warning")
			return
		
		# دریافت داده‌ها
		primary_df = mt5i.get_rates(symbol, Config.PRIMARY_TF, 300)
		entry_df = mt5i.get_rates(symbol, Config.ENTRY_TF, 100)
		confirm_df = mt5i.get_rates(symbol, Config.CONFIRM_TF, 50)
		
		if primary_df is None or primary_df.empty or entry_df is None or entry_df.empty or confirm_df is None or confirm_df.empty:
			log(f"داده‌ای برای {symbol} دریافت نشد", "warning")
			return
		
		# ارزیابی سیگنال
		signal = composite_engine.evaluate(symbol, primary_df, entry_df, confirm_df)
		
		# اجرای معامله در صورت وجود سیگنال مناسب
		if signal["decision"] in ["strong_entry", "conditional_entry"]:
			side = "buy" if signal["details"].get("trend") == "bull" else "sell"
			price = float(signal["price"]) or float(entry_df['close'].iloc[-1])
			sl = signal["suggested"]["sl"]
			tp = signal["suggested"]["tp"]
			
			if sl is None or tp is None:
				log(f"SL/TP نامعتبر برای {symbol}", "warning")
				return
				
			lot = risk_manager.calculate_position_size(symbol, Config.RISK["low_risk_pct"], price, sl)
			if lot <= 0:
				log(f"حجم معامله نامعتبر برای {symbol}", "warning")
				return
				
			order_manager.execute_order(symbol, side, lot, price, float(sl), float(tp))

	def _main_loop(self):
		"""حلقه اصلی مانیتورینگ"""
		while self.running:
			try:
				self.watchdog.beat()
				
				# پردازش موازی نمادها
				with ThreadPoolExecutor(max_workers=Config.THREAD_POOL.get("max_workers", 4)) as executor:
					futures = [executor.submit(self._process_symbol, symbol) for symbol in self.symbols]
					for _ in as_completed(futures):
						pass
				
				# استراحت بین چک‌ها
				time.sleep(60)
				
			except Exception as e:
				log(f"خطا در حلقه اصلی: {e}\n{traceback.format_exc()}", "error")
				time.sleep(30)

# -------------------------
# نقطه ورود برنامه
# -------------------------
if __name__ == "__main__":
	# در صورت عدم وجود کتابخانه‌های ضروری برای DataFrame، اجرای زنده را متوقف کن
	if not NUMPY_PANDAS:
		log("Pandas/Numpy مورد نیاز هستند. لطفاً نصب کنید.", "critical")
		sys.exit(1)
	
	monitor = TradingMonitor()
	
	try:
		monitor.start()
		
		# انتظار برای پایان کار در حالت کنسول
		while monitor.running:
			time.sleep(1)
			
	except KeyboardInterrupt:
		log("دریافت سیگنال توقف از کاربر", "info")
		monitor.stop()
	
	except Exception as e:
		log(f"خطای غیرمنتظره: {e}\n{traceback.format_exc()}", "critical")
		monitor.stop()