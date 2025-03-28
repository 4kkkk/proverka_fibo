import sys
import time
import json
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import dateutil.parser
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QDateEdit, QMessageBox, QComboBox,
                             QFileDialog, QTextEdit, QProgressBar, QGroupBox, QCheckBox, QSpinBox,
                             QDoubleSpinBox)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QTextCursor

class FiboAnalyzer:
    def __init__(self, percent_margin=0.5):
        self.base_url = "https://api.bybit.com"
        self.fibo_levels = {
            0: 0,
            0.236: 0.236,
            0.382: 0.382,
            0.5: 0.5,
            0.618: 0.618,
            0.786: 0.786,
            1: 1,
            1.618: 1.618,
            2.618: 2.618
        }
        # Устанавливаем триггерные уровни в правильном порядке от высшего к низшему
        self.trigger_levels = [0.382, 0.5, 0.618]
        # Для каждого триггерного уровня определяем цель отскока
        self.rebound_targets = {
            0.382: 0.5,  # От 0.382 отскок к 0.5
            0.5: 0.618,  # От 0.5 отскок к 0.618
            0.618: 0.786 # От 0.618 отскок к 0.786
        }
        self.percent_margin = percent_margin
        self.min_trend_length = 5  # Минимальная длина тренда
        # Добавляем атрибут для хранения часовых данных
        self.hourly_data_cache = {}

    def get_kline_data(self, symbol, start_time, end_time=None):
        endpoint = "/v5/market/kline"
        symbol_formatted = symbol
        interval_formatted = "D"

        if not end_time:
            end_time = int(time.time() * 1000)

        if isinstance(start_time, str):
            start_time = int(dateutil.parser.parse(start_time).timestamp() * 1000)

        params = {
            "category": "linear",
            "symbol": symbol_formatted,
            "interval": interval_formatted,
            "start": start_time,
            "end": end_time,
            "limit": 1000
        }

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()
            if 'retCode' in data and data['retCode'] != 0:
                return []

            if 'result' in data and 'list' in data['result']:
                return data['result']['list']
            else:
                return []
        except requests.exceptions.RequestException:
            return []

    def check_symbol_exists(self, symbol):
        endpoint = "/v5/market/instruments-info"
        params = {
            "category": "linear",
            "symbol": symbol
        }

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            data = response.json()

            if 'retCode' in data and data['retCode'] == 0:
                if 'result' in data and 'list' in data['result']:
                    return len(data['result']['list']) > 0
            return False
        except:
            return False

    def get_hourly_data(self, symbol, start_date, end_date=None):
        """Получает часовые свечи для анализа отскоков"""
        if isinstance(start_date, str):
            start_timestamp = int(dateutil.parser.parse(start_date).timestamp() * 1000)
        elif isinstance(start_date, pd.Timestamp):
            start_timestamp = int(start_date.timestamp() * 1000)
        else:
            start_timestamp = int(start_date.timestamp() * 1000)

        if end_date is None:
            end_timestamp = int(time.time() * 1000)
        elif isinstance(end_date, str):
            end_timestamp = int(dateutil.parser.parse(end_date).timestamp() * 1000)
        elif isinstance(end_date, pd.Timestamp):
            end_timestamp = int(end_date.timestamp() * 1000)
        else:
            end_timestamp = int(end_date.timestamp() * 1000)

        cache_key = f"{symbol}_{start_timestamp}_{end_timestamp}"
        if cache_key in self.hourly_data_cache:
            return self.hourly_data_cache[cache_key]

        # Получаем часовые свечи
        endpoint = "/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "60",  # Часовой интервал
            "start": start_timestamp,
            "end": end_timestamp,
            "limit": 1000
        }

        all_klines = []
        current_start = start_timestamp

        # Возможно, понадобится несколько запросов для получения всех данных
        while current_start < end_timestamp:
            try:
                params["start"] = current_start
                response = requests.get(f"{self.base_url}{endpoint}", params=params)
                response.raise_for_status()

                data = response.json()
                if 'retCode' in data and data['retCode'] != 0:
                    break

                if 'result' in data and 'list' in data['result'] and data['result']['list']:
                    klines = data['result']['list']
                    klines.reverse()  # Bybit возвращает от новых к старым
                    all_klines.extend(klines)

                    # Обновляем начало для следующего запроса
                    latest_timestamp = int(klines[0][0])
                    if latest_timestamp <= current_start:
                        break  # Предотвращаем зацикливание
                    current_start = latest_timestamp + 1
                else:
                    break

                # Пауза между запросами
                time.sleep(0.2)

                # Ограничение на количество запрашиваемых свечей
                if len(all_klines) >= 5000:
                    break

            except Exception as e:
                print(f"Ошибка при получении часовых данных: {str(e)}")
                break

        # Преобразуем в DataFrame
        if all_klines:
            all_klines.sort(key=lambda x: int(x[0]))
            hourly_df = self.prepare_dataframe(all_klines)
            self.hourly_data_cache[cache_key] = hourly_df
            return hourly_df
        else:
            return pd.DataFrame()





    def get_complete_data(self, symbol, start_date):
        if isinstance(start_date, str):
            start_timestamp = int(dateutil.parser.parse(start_date).timestamp() * 1000)
        else:
            start_timestamp = int(start_date.timestamp() * 1000)

        current_timestamp = int(time.time() * 1000)

        all_klines = []
        last_timestamp = start_timestamp
        max_requests = 5

        for i in range(max_requests):
            klines = self.get_kline_data(symbol, last_timestamp, current_timestamp)

            if not klines:
                break

            klines.reverse()
            all_klines.extend(klines)

            if len(klines) < 1000 or len(all_klines) > 3000:
                if len(all_klines) > 3000:
                    all_klines = all_klines[-3000:]
                break

            last_timestamp = int(klines[-1][0]) + 1

            if last_timestamp >= current_timestamp:
                break

            time.sleep(0.2)

        if not all_klines:
            return []

        all_klines.sort(key=lambda x: int(x[0]))
        return all_klines

    def prepare_dataframe(self, klines):
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        df.rename(columns={'timestamp': 'date'}, inplace=True)  # Переименуем для совместимости с нашей логикой
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        return df

    def find_uptrend(self, df):
        """
        Находит последний восходящий тренд на графике.
        Возвращает индексы начала и конца тренда, минимум и максимум
        """
        if df.empty:
            print("DataFrame пустой")
            return None, None, None, None, None

        # Определим локальные минимумы и максимумы
        df['min_idx'] = self._find_local_minima(df['low'], window=5)
        df['max_idx'] = self._find_local_maxima(df['high'], window=5)

        # Выводим количество найденных минимумов и максимумов
        min_count = df['min_idx'].sum()
        max_count = df['max_idx'].sum()
        print(f"Найдено локальных минимумов: {min_count}")
        print(f"Найдено локальных максимумов: {max_count}")

        # Собираем все локальные минимумы и сортируем их от новых к старым
        min_candidates = []
        for i in range(len(df)):
            if df['min_idx'].iloc[i]:
                min_candidates.append(i)

        # Сортируем минимумы от более новых (больший индекс) к более старым
        min_candidates.sort(reverse=True)

        # Если нет минимумов, выходим
        if not min_candidates:
            print("Не найдено локальных минимумов")
            return None, None, None, None, None

        print(f"Начинаем анализ {len(min_candidates)} минимумов")

        # Проверяем каждый минимум, начиная с самого последнего (ближайшего к текущей дате)
        for min_idx in min_candidates:
            # Проверяем, что есть достаточно свечей после этого минимума
            if len(df) - min_idx < self.min_trend_length:
                print(f"Минимум на позиции {min_idx}: недостаточно свечей после него")
                continue

            min_price = df['low'].iloc[min_idx]
            min_date = df['date'].iloc[min_idx].strftime('%Y-%m-%d')
            print(f"Анализ минимума на позиции {min_idx} (дата: {min_date}, цена: {min_price})")

            # Ищем максимум от минимума до конца графика
            max_price = -1
            max_idx = -1

            # Просматриваем все свечи после минимума
            for i in range(min_idx + 1, len(df)):
                curr_price = df['high'].iloc[i]
                if curr_price > max_price:
                    max_price = curr_price
                    max_idx = i

            max_date = df['date'].iloc[max_idx].strftime('%Y-%m-%d')
            print(f"  Найден максимум на позиции {max_idx} (дата: {max_date}, цена: {max_price})")

            # Проверяем, соблюдается ли минимальная длина тренда
            if max_idx - min_idx >= self.min_trend_length:
                # Проверяем, действительно ли это восходящий тренд
                trend_segment = df.iloc[min_idx:max_idx+1]
                is_up = self._is_uptrend(trend_segment)
                print(f"  Длина тренда: {max_idx - min_idx} свечей, восходящий: {is_up}")

                if is_up:
                    # Проверяем, не было ли значительного падения после этого минимума
                    valid_trend = True

                    # Проверяем, не опускалась ли цена ниже минимума тренда после максимума
                    if max_idx < len(df) - 1:  # Если есть свечи после максимума
                        post_max_segment = df.iloc[max_idx+1:]
                        if len(post_max_segment) > 0 and post_max_segment['low'].min() < min_price:
                            # Цена опускалась ниже минимума - этот тренд "отменен"
                            valid_trend = False
                            print(f"  Тренд отменен: цена опускалась ниже минимума после достижения максимума")

                    if valid_trend:
                        print(f"  Найден валидный тренд!")

                        # Ищем дополнительно 40 свечей ПЕРЕД найденным трендом для поиска более низкой цены
                        additional_min_price = min_price
                        additional_min_idx = min_idx

                        # Определяем начало периода для анализа (не выходя за границы DataFrame)
                        pre_trend_start = max(0, min_idx - 50)

                        if pre_trend_start < min_idx:  # Если есть свечи перед минимумом
                            pre_trend_segment = df.iloc[pre_trend_start:min_idx]

                            # Ищем новый минимум в этих свечах
                            if not pre_trend_segment.empty:
                                new_min_price = pre_trend_segment['low'].min()

                                if new_min_price < min_price:
                                    # Находим индекс минимума в вырезанном сегменте
                                    new_min_relative_idx = pre_trend_segment['low'].values.argmin()
                                    new_min_idx = pre_trend_start + new_min_relative_idx

                                    new_min_date = df['date'].iloc[new_min_idx].strftime('%Y-%m-%d')
                                    print(f"  Найдена более низкая цена {new_min_price} на позиции {new_min_idx} (дата: {new_min_date})")

                                    # Проверяем, есть ли после этой цены восходящий тренд до текущего максимума
                                    if new_min_idx < min_idx:  # Убедимся, что новый минимум раньше текущего
                                        # Проверяем сегмент от нового минимума до текущего максимума
                                        extended_segment = df.iloc[new_min_idx:max_idx+1]
                                        if self._is_uptrend(extended_segment):
                                            print(f"  Найден расширенный восходящий тренд от {new_min_price} до {max_price}")
                                            # Используем более низкую цену для расчета Фибоначчи
                                            additional_min_price = new_min_price
                                            additional_min_idx = new_min_idx

                                            # Рассчитаем процент роста для расширенного тренда
                                            growth_percent = ((max_price / additional_min_price) - 1) * 100
                                            print(f"  Процент роста: {growth_percent:.2f}%")

                                            return additional_min_idx, max_idx, additional_min_price, max_price, growth_percent

                        # Если не нашли расширенный тренд, возвращаем оригинальный
                        growth_percent = ((max_price / min_price) - 1) * 100
                        print(f"  Процент роста: {growth_percent:.2f}%")

                        return min_idx, max_idx, min_price, max_price, growth_percent
                else:
                    print(f"  Не является восходящим трендом по линейной регрессии")
            else:
                print(f"  Недостаточная длина тренда: {max_idx - min_idx} (минимум {self.min_trend_length})")

        # Если не нашли ни одного подходящего тренда
        print("Не найдено подходящих трендов")
        return None, None, None, None, None

    def _find_local_minima(self, series, window=5):
        """Находит локальные минимумы"""
        local_min = np.zeros(len(series), dtype=bool)

        for i in range(window, len(series) - window):
            if series.iloc[i] == min(series.iloc[i-window:i+window+1]):
                local_min[i] = True

        return local_min

    def _find_local_maxima(self, series, window=5):
            """Находит локальные максимумы"""
            local_max = np.zeros(len(series), dtype=bool)

            for i in range(window, len(series) - window):
                if series.iloc[i] == max(series.iloc[i-window:i+window+1]):
                    local_max[i] = True

            return local_max

    def _is_uptrend(self, segment):
        """Проверяет, является ли сегмент восходящим трендом"""
        # Проверяем процентное изменение от начала до конца
        start_price = segment['close'].iloc[0]
        end_price = segment['close'].iloc[-1]
        percent_change = (end_price / start_price - 1) * 100

        # Если изменение более 3%, считаем это трендом
        if percent_change > 3:
            return True

        # Также используем линейную регрессию для более точной проверки
        x = np.arange(len(segment))
        y = segment['close'].values
        slope = np.polyfit(x, y, 1)[0]

        # Положительный наклон означает восходящий тренд
        return slope > 0

    def calculate_fibo_levels(self, df, logarithmic=True):
        """
        Автоматически находит тренд и рассчитывает уровни Фибоначчи
        Возвращает словарь с уровнями и информацию о тренде
        """
        if df.empty:
            raise ValueError("Получен пустой DataFrame. Проверьте параметры запроса.")

        # Находим восходящий тренд
        min_idx, max_idx, min_price, max_price, growth_percent = self.find_uptrend(df)

        if min_idx is None:
            raise ValueError("Не удалось найти подходящий восходящий тренд")

        # Рассчитываем уровни Фибоначчи
        fibo_values = {}

        if logarithmic and (min_price <= 0 or max_price <= 0):
            logarithmic = False

        if logarithmic:
            log_max = np.log(max_price)
            log_min = np.log(min_price)
            log_diff = log_max - log_min

            for level, ratio in self.fibo_levels.items():
                log_level = log_min + log_diff * ratio
                fibo_values[level] = np.exp(log_level)
        else:
            diff = max_price - min_price

            for level, ratio in self.fibo_levels.items():
                fibo_values[level] = min_price + diff * ratio

        return fibo_values, max_price, min_price, max_idx, min_idx, growth_percent

    def print_fibo_levels_with_margin(self, fibo_values):
        print("\n=== Уровни Фибоначчи с учетом Разбега ({}%) ===".format(self.percent_margin))

        for level in [0.382, 0.5, 0.618, 0.786]:
            if level in fibo_values:
                level_price = fibo_values[level]
                # Рассчитываем уровень с учетом Разбега для проверки достижения (верхняя граница)
                margin_for_reach = level_price * (self.percent_margin / 100)
                level_with_margin_reach = level_price + margin_for_reach

                # Рассчитываем уровень с учетом Разбега для проверки отскока (нижняя граница)
                margin_for_rebound = level_price * (self.percent_margin / 100)
                level_with_margin_rebound = level_price - margin_for_rebound

                print(f"Уровень {level}: {level_price}")
                print(f"  - С учетом Разбега для достижения (верхняя граница): {level_with_margin_reach}")
                print(f"  - С учетом Разбега для отскока (нижняя граница): {level_with_margin_rebound}")

    def is_level_reached(self, price, level_price):
        """
        Проверяет, достигла ли цена указанного уровня с учетом разбега
        price: одиночное значение или серия Pandas
        """
        if self.percent_margin == 0:
            return price <= level_price

        margin = level_price * (self.percent_margin / 100)
        boundary = level_price + margin

        # Если подается серия, печатать только для отладки
        if isinstance(price, pd.Series):
            print(f"Проверка уровня: цена (серия длиной {len(price)}), уровень {level_price}, граница {boundary}")
            return price <= boundary
        else:
            # Для отдельных значений можно печатать подробнее
            print(f"Проверка уровня: цена {price}, уровень {level_price}, граница {boundary}")
            return price <= boundary

    def is_level_reached_vectorized(self, prices, level_price):
        """Векторизованная версия проверки достижения уровня"""
        if self.percent_margin == 0:
            return prices <= level_price

        margin = level_price * (self.percent_margin / 100)
        return prices <= level_price + margin
    def is_rebound_to_level(self, price, level_price):
        if self.percent_margin == 0:
            return price >= level_price
        margin = level_price * (self.percent_margin / 100)
        return price >= level_price - margin

    def analyze_strategy(self, df, fibo_values, max_idx, symbol):
        """Анализирует стратегию с использованием гибридного подхода"""
        df_after_max = df.iloc[max_idx:]

        print(f"\nАнализ тренда на дневном графике:")
        print(f"Максимум найден на позиции {max_idx}, дата: {df.iloc[max_idx]['date']}")
        for level in [0.382, 0.5, 0.618, 0.786]:
            if level in fibo_values:
                print(f"Уровень {level}: {fibo_values[level]}")

        self.print_fibo_levels_with_margin(fibo_values)

        # Проверяем, какие триггерные уровни были достигнуты на дневном графике
        reached_levels = {}
        hit_dates = {}  # Словарь для хранения дат первого касания уровней

        for level in self.trigger_levels:
            level_price = fibo_values[level]
            hit_frames = df_after_max[self.is_level_reached_vectorized(df_after_max['low'], level_price)]

            if not hit_frames.empty:
                reached_levels[level] = True
                first_hit_date = hit_frames.iloc[0]['date']
                hit_dates[level] = first_hit_date
                print(f"Уровень {level} ({level_price}) достигнут на дневном графике {first_hit_date}")
                # Добавьте эти строки
                margin = level_price * (self.percent_margin / 100)
                level_with_margin = level_price + margin
                print(f"  - С учетом Разбега ({self.percent_margin}%): до {level_with_margin}")
            else:
                reached_levels[level] = False
                print(f"Уровень {level} ({level_price}) не достигнут на дневном графике")
                # Добавьте эти строки
                margin = level_price * (self.percent_margin / 100)
                level_with_margin = level_price + margin
                print(f"  - С учетом Разбега ({self.percent_margin}%): до {level_with_margin}")


# Проверяем стоп-лосс на дневном графике
        stop_loss_triggered = any(df_after_max['low'] <= fibo_values[0] + (fibo_values[0.236] - fibo_values[0]) * 0.1)

        # Анализируем отскоки на часовом графике
        rebounds = {}

        for level in self.trigger_levels:
            if reached_levels[level]:
                target_level = self.rebound_targets[level]
                target_price = fibo_values[target_level]

                # Определяем период для анализа часовых данных
                hit_date = hit_dates[level]
                # Берем 3 дня до первого касания для контекста
                start_date = hit_date - pd.Timedelta(days=3)
                # И до текущего времени или максимум 14 дней после касания
                end_date = min(pd.Timestamp.now(), hit_date + pd.Timedelta(days=14))

                print(f"\nПолучаем часовые данные с {start_date} по {end_date} для анализа отскока от уровня {level}")

                # Получаем часовые данные
                hourly_df = self.get_hourly_data(symbol, start_date, end_date)

                if hourly_df.empty:
                    print(f"Не удалось получить часовые данные для {symbol}")
                    rebounds[level] = False
                    continue

                # Находим первое касание уровня на часовом графике
                level_price = fibo_values[level]  # Получаем цену текущего уровня
                hourly_hit_frames = hourly_df[self.is_level_reached_vectorized(hourly_df['low'], level_price)]
                if not hourly_hit_frames.empty:
                    first_hourly_hit = hourly_hit_frames.iloc[0]
                    print(f"Первое касание уровня {level} ({level_price}) на часовом графике: {first_hourly_hit['date']}")
                    print(f"Цена касания: {first_hourly_hit['low']}")
                    print(f"Разница: {abs(level_price - first_hourly_hit['low']) / level_price * 100:.4f}%")
                    print(f"Макс. допустимая разница (Разбег): {self.percent_margin}%")
                if hourly_hit_frames.empty:
                    print(f"На часовом графике не найдено касание уровня {level}")
                    rebounds[level] = False
                    continue

                first_hourly_hit = hourly_hit_frames.iloc[0]
                first_hourly_hit_date = first_hourly_hit['date']
                print(f"Первое касание уровня {level} на часовом графике: {first_hourly_hit_date}")

                # Выбираем данные после первого касания на часовом графике
                hourly_after_hit = hourly_df[hourly_df['date'] > first_hourly_hit_date]

                # Анализируем отскок к целевому уровню
                hourly_rebound_frames = hourly_after_hit[self.is_rebound_to_level_vectorized(hourly_after_hit['high'], target_price)]

                if not hourly_rebound_frames.empty:
                    rebound_date = hourly_rebound_frames.iloc[0]['date']
                    print(f"Найден отскок к уровню {target_level} ({target_price}) на часовом графике: {rebound_date}")
                    margin = target_price * (self.percent_margin / 100)
                    target_with_margin = target_price - margin
                    print(f"  - С учетом Разбега ({self.percent_margin}%): от {target_with_margin}")
                    # Проверяем, не является ли отскок частью того же движения
                    # Должно пройти минимум 3 часа между касанием и отскоком
                    time_diff = (rebound_date - first_hourly_hit_date).total_seconds() / 3600

                    if time_diff < 3:
                        print(f"Отскок произошел слишком быстро ({time_diff} часов) - это не настоящий отскок")
                        rebounds[level] = False
                    else:
                        print(f"Подтвержденный отскок через {time_diff} часов")
                        rebounds[level] = True
                else:
                    print(f"Отскок к уровню {target_level} не найден на часовом графике")
                    rebounds[level] = False
            else:
                rebounds[level] = False

        # Стратегия отработала, если хотя бы один уровень достиг отскока
        strategy_worked = any(rebounds.values())

        return {
            'reached_levels': reached_levels,
            'rebounds': rebounds,
            'stop_loss_triggered': stop_loss_triggered,
            'strategy_worked': strategy_worked
        }

    # Вспомогательные методы для векторизованного анализа
    def is_level_reached_vectorized(self, prices, level_price):
        """Векторизованная версия проверки достижения уровня"""
        if self.percent_margin == 0:
            return prices <= level_price

        margin = level_price * (self.percent_margin / 100)
        # Проверяем, что цена не более чем на margin выше уровня
        return prices <= level_price + margin

    def is_rebound_to_level_vectorized(self, prices, level_price):
        """Векторизованная версия проверки отскока"""
        if self.percent_margin == 0:
            return prices >= level_price
        margin = level_price * (self.percent_margin / 100)
        return prices >= level_price - margin

def get_price_precision(price):
    if price == 0:
        return 6
    precision = 0
    temp_price = price
    while temp_price < 1 and temp_price > 0:
        temp_price *= 10
        precision += 1
    return precision + 2

def parse_coin_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        coins_section = re.search(r'Монеты, которые выросли.*?:(.*?)(?:$|\n\n)', content, re.DOTALL)
        if not coins_section:
            return []

        coins_text = coins_section.group(1)

        # Обновленный паттерн для формата "10000000AIDOGEUSDT: 43.06% (Возраст: 386 дней)"
        coin_pattern = r'(\w+USDT): ([\d\.]+)%'
        coins = re.findall(coin_pattern, coins_text)

        return [coin[0] for coin in coins]
    except Exception as e:
        print(f"Ошибка при парсинге файла: {str(e)}")
        return []


class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, coin_list, start_date, percent_margin=0.5, use_logarithmic=True):
        super().__init__()
        self.coin_list = coin_list
        self.start_date = start_date
        self.percent_margin = percent_margin
        self.use_logarithmic = use_logarithmic
        self.analyzer = FiboAnalyzer(percent_margin=percent_margin)
        self.results = {}
        self.errors = {}
        self.stop_flag = False

    def run(self):
        try:
            for i, symbol in enumerate(self.coin_list):
                if self.stop_flag:
                    self.finished_signal.emit("Анализ прерван пользователем")
                    return

                progress = int(i / len(self.coin_list) * 100)
                self.progress_signal.emit(progress, f"Анализ {symbol}... ({i+1}/{len(self.coin_list)})")

                try:
                    if not self.analyzer.check_symbol_exists(symbol):
                        self.errors[symbol] = "Символ не найден на Bybit"
                        continue

                    # Получаем исторические данные для всего доступного периода
                    # Используем фиксированную дату далеко в прошлом или дату из настроек, если она задана
                    far_past_date = self.start_date if self.start_date else datetime.now() - timedelta(days=180)

                    klines = self.analyzer.get_complete_data(symbol, far_past_date)
                    if not klines:
                        self.errors[symbol] = "Не удалось получить данные"
                        continue

                    df = self.analyzer.prepare_dataframe(klines)
                    if df.empty:
                        self.errors[symbol] = "Пустой набор данных"
                        continue

                    # Находим тренд и рассчитываем уровни Фибоначчи
                    fibo_values, max_price, min_price, max_idx, min_idx, growth_percent = self.analyzer.calculate_fibo_levels(
                        df, logarithmic=self.use_logarithmic
                    )

                    # Анализируем стратегию на основе найденного тренда,
                    # передаем символ для запроса часовых данных для анализа отскоков
                    analysis_result = self.analyzer.analyze_strategy(df, fibo_values, max_idx, symbol)

                    price_precision = get_price_precision(max_price)

                    self.results[symbol] = {
                        'max_price': max_price,
                        'max_date': df.iloc[max_idx]['date'].strftime('%Y-%m-%d %H:%M'),
                        'min_price': min_price,
                        'min_date': df.iloc[min_idx]['date'].strftime('%Y-%m-%d %H:%M'),
                        'growth_percent': growth_percent,
                        'fibo_values': {k: round(v, price_precision) for k, v in fibo_values.items()},
                        'analysis': analysis_result,
                        'price_precision': price_precision
                    }

                except Exception as e:
                    self.errors[symbol] = f"Ошибка анализа: {str(e)}"
                    import traceback
                    print(f"Ошибка для {symbol}: {str(e)}")
                    print(traceback.format_exc())

                if i < len(self.coin_list) - 1:
                    time.sleep(0.2)

            self.result_signal.emit(self.results)

            success_count = len(self.results)
            total_count = len(self.coin_list)
            error_count = len(self.errors)

            finish_msg = f"Анализ завершен. Успешно: {success_count}/{total_count}."
            if error_count > 0:
                finish_msg += f" Ошибок: {error_count}."

            self.finished_signal.emit(finish_msg)

        except Exception as e:
            self.error_signal.emit(f"Критическая ошибка: {str(e)}")
            import traceback
            print(f"Критическая ошибка: {str(e)}")
            print(traceback.format_exc())

def stop(self):
    self.stop_flag = True


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.analyzer = FiboAnalyzer()
        self.coin_list = []
        self.results = {}
        self.worker = None

        self.setWindowTitle('Анализатор Фибоначчи для множества монет')
        self.setGeometry(100, 100, 1000, 800)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        file_group = QGroupBox("Файл с монетами")
        file_layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("Выберите файл с монетами...")

        self.browse_button = QPushButton("Обзор...")
        self.browse_button.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_path_edit, 3)
        file_layout.addWidget(self.browse_button, 1)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        params_group = QGroupBox("Параметры анализа")
        params_layout = QHBoxLayout()

        date_label = QLabel("Не анализировать данные ранее (Москва UTC+3):")
        self.date_edit = QDateEdit()
        default_date = QDate.currentDate().addMonths(-3)
        self.date_edit.setDate(default_date)
        self.date_edit.setMaximumDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)

        margin_label = QLabel("Разбег (%):")
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 5.0)
        self.margin_spin.setValue(0.5)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setSuffix("%")

        self.log_scale_check = QCheckBox("Логарифмическая шкала")
        self.log_scale_check.setChecked(True)

        params_layout.addWidget(date_label)
        params_layout.addWidget(self.date_edit)
        params_layout.addWidget(margin_label)
        params_layout.addWidget(self.margin_spin)
        params_layout.addWidget(self.log_scale_check)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        control_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Начать анализ")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)

        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)

        control_layout.addWidget(self.analyze_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)

        main_layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Готов к работе")

        progress_layout.addWidget(self.progress_bar, 3)
        progress_layout.addWidget(self.status_label, 1)

        main_layout.addLayout(progress_layout)

        result_group = QGroupBox("Результаты анализа")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Courier New", 10))

        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)

        main_layout.addWidget(result_group, 1)

        self.load_settings()

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с монетами", "",
            "Текстовые файлы (*.txt);;Все файлы (*)", options=options
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_coins_from_file(file_path)

    def load_coins_from_file(self, file_path):
        try:
            self.coin_list = parse_coin_file(file_path)
            if self.coin_list:
                self.status_label.setText(f"Загружено {len(self.coin_list)} монет")
                self.analyze_button.setEnabled(True)

                self.result_text.clear()
                self.result_text.append(f"Загружены монеты из файла: {file_path}\n")
                self.result_text.append(f"Всего монет: {len(self.coin_list)}\n\n")
                self.result_text.append("Список монет для анализа:")

                coins_per_row = 4
                for i in range(0, len(self.coin_list), coins_per_row):
                    chunk = self.coin_list[i:i+coins_per_row]
                    self.result_text.append("  ".join(chunk))
            else:
                self.status_label.setText("Ошибка: не удалось найти монеты в файле")
                QMessageBox.warning(self, "Ошибка", "Не удалось найти монеты в файле.\nПроверьте формат файла.")
        except Exception as e:
            self.status_label.setText(f"Ошибка загрузки файла: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при чтении файла: {str(e)}")

    def start_analysis(self):
        if not self.coin_list:
            QMessageBox.warning(self, "Предупреждение", "Список монет пуст. Загрузите файл с монетами.")
            return

        # Получаем дату в московском времени и преобразуем в UTC
        start_date = self.date_edit.date().toString("yyyy-MM-dd") + " 00:00:00"
        moscow_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        utc_date = moscow_date - timedelta(hours=3)  # Переводим из Москвы в UTC

        percent_margin = self.margin_spin.value()
        use_logarithmic = self.log_scale_check.isChecked()
        self.worker = AnalysisWorker(self.coin_list, utc_date, percent_margin, use_logarithmic)
        self.results = {}
        self.result_text.clear()
        self.result_text.append(f"Начало анализа от {start_date} (Москва UTC+3), разбег: {percent_margin}%\n")

        self.worker = AnalysisWorker(self.coin_list, utc_date, percent_margin, use_logarithmic)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.result_signal.connect(self.process_results)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.handle_error)

        self.analyze_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker.start()

        self.save_settings()

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Останавливаем анализ...")
            self.stop_button.setEnabled(False)

    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

        if "Анализ" in message:
            self.result_text.append(message)
            self.result_text.moveCursor(QTextCursor.End)

    def process_results(self, results):
        self.results = results

    def analysis_finished(self, message):
        self.status_label.setText(message)
        self.progress_bar.setValue(100)

        self.analyze_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)

        self.display_results()

    def handle_error(self, message):
        self.status_label.setText(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка во время анализа:\n{message}")

        self.analyze_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def display_results(self):
        if not self.results:
            self.result_text.append("\nАнализ не дал результатов.")
            return

        self.result_text.clear()

        total_coins = len(self.results)
        strategy_worked = sum(1 for r in self.results.values() if r['analysis']['strategy_worked'])
        stoploss_triggered = sum(1 for r in self.results.values() if r['analysis']['stop_loss_triggered'])
        pending_coins = total_coins - strategy_worked - stoploss_triggered

        success_rate = strategy_worked / total_coins * 100 if total_coins > 0 else 0

        self.result_text.append(f"=== Результаты анализа ===\n")
        self.result_text.append(f"Дата начала: {self.date_edit.date().toString('yyyy-MM-dd')} (Москва UTC+3)")
        self.result_text.append(f"Разбег: {self.margin_spin.value()}%\n")
        self.result_text.append(f"Всего проанализировано монет: {total_coins}")
        self.result_text.append(f"Стратегия сработала: {strategy_worked} ({success_rate:.2f}%)")
        self.result_text.append(f"Стоп-лосс сработал: {stoploss_triggered}")
        self.result_text.append(f"В процессе ожидания: {pending_coins}\n")

        # Группируем монеты по результату
        worked_coins = [symbol for symbol, result in self.results.items()
                        if result['analysis']['strategy_worked']]  # Теперь сюда попадут и монеты со стоп-лоссом

        # Только монеты где не отработал ни один уровень и был стоп-лосс
        stoploss_coins = [symbol for symbol, result in self.results.items()
                          if not result['analysis']['strategy_worked'] and result['analysis']['stop_loss_triggered']]

        # Монеты в ожидании - не отработали и без стоп-лосса
        pending_coins = [symbol for symbol, result in self.results.items()
                         if not result['analysis']['strategy_worked'] and not result['analysis']['stop_loss_triggered']]


        # Сортируем внутри каждой группы
        worked_coins.sort()
        pending_coins.sort()
        stoploss_coins.sort()

        # Объединяем в нужном порядке
        sorted_coins = worked_coins + pending_coins + stoploss_coins

        # Отображаем монеты по группам
        if worked_coins:
            self.result_text.append("\n=== МОНЕТЫ, ГДЕ СТРАТЕГИЯ ОТРАБОТАЛА ===")
            for symbol in worked_coins:
                self.display_coin_result(symbol)

        if pending_coins:
            self.result_text.append("\n=== МОНЕТЫ, ГДЕ ОЖИДАЕТСЯ ОТРАБОТКА ===")
            for symbol in pending_coins:
                self.display_coin_result(symbol)

        if stoploss_coins:
            self.result_text.append("\n=== МОНЕТЫ, ГДЕ СРАБОТАЛ СТОП-ЛОСС ===")
            for symbol in stoploss_coins:
                self.display_coin_result(symbol)

        self.result_text.moveCursor(QTextCursor.Start)

    def display_coin_result(self, symbol):
        result = self.results[symbol]

        # Преобразуем UTC в московское время (UTC+3)
        max_date_utc = datetime.strptime(result['max_date'], '%Y-%m-%d %H:%M')
        max_date_moscow = max_date_utc + timedelta(hours=3)
        max_date_str = max_date_moscow.strftime('%Y-%m-%d %H:%M')

        min_date_utc = datetime.strptime(result['min_date'], '%Y-%m-%d %H:%M')
        min_date_moscow = min_date_utc + timedelta(hours=3)
        min_date_str = min_date_moscow.strftime('%Y-%m-%d %H:%M')

        if result['analysis']['strategy_worked']:
            strategy_text = "✅ Стратегия ОТРАБОТАЛА"
        else:
            if result['analysis']['stop_loss_triggered']:
                strategy_text = "❌ Сработал СТОП-ЛОСС"
            else:
                strategy_text = "⏳ Ожидается отработка"

        self.result_text.append(f"\n-- {symbol} --")
        self.result_text.append(f"Тренд: с {min_date_str} по {max_date_str} (Москва UTC+3)")
        self.result_text.append(f"Минимальная цена: {result['min_price']:.{result['price_precision']}f}")
        self.result_text.append(f"Максимальная цена: {result['max_price']:.{result['price_precision']}f}")
        self.result_text.append(f"Рост: {result['growth_percent']:.2f}%")
        self.result_text.append(strategy_text)

        for level in self.analyzer.trigger_levels:
            if level in result['fibo_values'] and level in result['analysis']['reached_levels']:
                level_price = result['fibo_values'][level]
                if result['analysis']['reached_levels'][level]:
                    if result['analysis']['rebounds'].get(level, False):
                        self.result_text.append(f"✅ Уровень {level} ({level_price:.{result['price_precision']}f}) достигнут и был отскок")
                    else:
                        self.result_text.append(f"⚠️ Уровень {level} ({level_price:.{result['price_precision']}f}) достигнут, но не было отскока")
                else:
                    self.result_text.append(f"⚫ Уровень {level} ({level_price:.{result['price_precision']}f}) не был достигнут")

    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для сохранения.")
            return

        options = QFileDialog.Options()
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить результаты",
            f"fibo_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            "Текстовые файлы (*.txt);;Excel файлы (*.xlsx);;Все файлы (*)",
            options=options
        )

        if not file_name:
            return

        try:
            if selected_filter == "Excel файлы (*.xlsx)":
                if not file_name.endswith('.xlsx'):
                    file_name += '.xlsx'
                self.save_to_excel(file_name)
            else:
                if not file_name.endswith('.txt'):
                    file_name += '.txt'
                self.save_to_text(file_name)

            QMessageBox.information(self, "Успех", f"Результаты сохранены в файл:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты:\n{str(e)}")

    def save_to_text(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Анализ Фибоначчи от {self.date_edit.date().toString('yyyy-MM-dd')} (Москва UTC+3)\n")
            f.write(f"Разбег уровней: {self.margin_spin.value()}%\n\n")
            f.write(self.result_text.toPlainText())

    def save_to_excel(self, file_name):
        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import PatternFill, Font

            data = []

            # Дата анализа и процент разбега для заголовка
            analysis_date = self.date_edit.date().toString('yyyy-MM-dd')
            margin_percent = self.margin_spin.value()

            # Группируем монеты по результату
            worked_coins = [symbol for symbol, result in self.results.items()
                            if result['analysis']['strategy_worked']]
            pending_coins = [symbol for symbol, result in self.results.items()
                             if not result['analysis']['strategy_worked'] and not result['analysis']['stop_loss_triggered']]
            stoploss_coins = [symbol for symbol, result in self.results.items()
                              if result['analysis']['stop_loss_triggered']]

            # Сортируем внутри каждой группы
            worked_coins.sort()
            pending_coins.sort()
            stoploss_coins.sort()

            # Объединяем в нужном порядке
            sorted_coins = worked_coins + pending_coins + stoploss_coins

            for symbol in sorted_coins:
                result = self.results[symbol]

                # Определяем статус стратегии
                if result['analysis']['strategy_worked']:
                    strategy_status = "Отработала"
                    group = "Отработала"
                elif result['analysis']['stop_loss_triggered']:
                    strategy_status = "Стоп-лосс"
                    group = "Стоп-лосс"
                else:
                    strategy_status = "Ожидание"
                    group = "Ожидание"

                row = {
                    'Группа': group,
                    'Символ': symbol,
                    'Мин. цена': result['min_price'],
                    'Дата минимума': result['min_date'],
                    'Макс. цена': result['max_price'],
                    'Дата максимума': result['max_date'],
                    'Рост %': result['growth_percent'],
                    'Статус': strategy_status,
                }

                # Добавляем информацию о уровнях
                for level in self.analyzer.trigger_levels:
                    if level in result['fibo_values'] and level in result['analysis']['reached_levels']:
                        level_price = result['fibo_values'][level]
                        row[f'Уровень {level}'] = level_price

                        if result['analysis']['reached_levels'][level]:
                            row[f'{level} достигнут'] = 'Да'
                            if result['analysis']['rebounds'].get(level, False):
                                row[f'{level} отработал'] = 'Да'
                            else:
                                row[f'{level} отработал'] = 'Нет'
                        else:
                            row[f'{level} достигнут'] = 'Нет'
                            row[f'{level} отработал'] = 'Нет'

                data.append(row)

            # Создаем DataFrame и сохраняем в Excel
            df = pd.DataFrame(data)

            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Результаты', index=False)

                # Применяем форматирование
                workbook = writer.book
                worksheet = writer.sheets['Результаты']

                # Добавляем информацию о параметрах анализа
                stats_sheet = workbook.create_sheet(title="Информация")

                stats_sheet['A1'] = "Параметры анализа"
                stats_sheet['A1'].font = Font(bold=True, size=14)

                stats_sheet['A3'] = "Разбег уровней:"
                stats_sheet['B3'] = f"{margin_percent}%"

                stats_sheet['A4'] = "Логарифмическая шкала:"
                stats_sheet['B4'] = "Да" if self.log_scale_check.isChecked() else "Нет"

                stats_sheet['A6'] = "Всего монет:"
                stats_sheet['B6'] = len(self.results)

                stats_sheet['A7'] = "Стратегия отработала:"
                stats_sheet['B7'] = len(worked_coins)
                stats_sheet['C7'] = f"{len(worked_coins)/len(self.results)*100:.2f}%" if self.results else "0%"

                stats_sheet['A8'] = "В ожидании отработки:"
                stats_sheet['B8'] = len(pending_coins)
                stats_sheet['C8'] = f"{len(pending_coins)/len(self.results)*100:.2f}%" if self.results else "0%"

                stats_sheet['A9'] = "Сработал стоп-лосс:"
                stats_sheet['B9'] = len(stoploss_coins)
                stats_sheet['C9'] = f"{len(stoploss_coins)/len(self.results)*100:.2f}%" if self.results else "0%"

                # Автонастройка ширины колонок
                for sheet in workbook.worksheets:
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter

                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass

                        adjusted_width = (max_length + 2)
                        sheet.column_dimensions[column_letter].width = adjusted_width

        except ImportError:
            QMessageBox.warning(self, "Предупреждение",
                                "Для сохранения в Excel требуются библиотеки pandas и openpyxl.\n"
                                "Результаты будут сохранены в текстовом формате.")
            self.save_to_text(file_name.replace('.xlsx', '.txt'))

    def load_settings(self):
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fibo_batch_settings.json')

        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)

                if 'margin' in settings:
                    self.margin_spin.setValue(settings['margin'])

                if 'start_date' in settings:
                    date = QDate.fromString(settings['start_date'], "yyyy-MM-dd")
                    if date.isValid():
                        self.date_edit.setDate(date)

                if 'log_scale' in settings:
                    self.log_scale_check.setChecked(settings['log_scale'])

                if 'last_file' in settings and os.path.exists(settings['last_file']):
                    self.file_path_edit.setText(settings['last_file'])
                    self.load_coins_from_file(settings['last_file'])

            except Exception as e:
                print(f"Ошибка при загрузке настроек: {str(e)}")

    def save_settings(self):
        settings = {
            'margin': self.margin_spin.value(),
            'start_date': self.date_edit.date().toString("yyyy-MM-dd"),
            'log_scale': self.log_scale_check.isChecked(),
            'last_file': self.file_path_edit.text() if os.path.exists(self.file_path_edit.text()) else ""
        }

        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fibo_batch_settings.json')

        try:
            with open(settings_path, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Ошибка при сохранении настроек: {str(e)}")

    def closeEvent(self, event):
        self.save_settings()

        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Подтверждение',
                                         'Анализ все еще выполняется. Вы уверены, что хотите выйти?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.worker.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())