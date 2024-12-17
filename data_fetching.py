import ccxt 
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from datetime import timedelta
import concurrent.futures
import time

class BinanceDataExtractor():
    def __init__(self, start_time, start_ts, end_time, interval_minutes=30):
        self.start_time = start_time
        self.end_time = end_time
        self.interval_minutes = interval_minutes
        self.binance = ccxt.binance()
        self.trading_pairs = self.get_trading_pairs()
        self.start_ts = start_ts
        
        
    def get_trading_pairs(self):
        url = 'https://coinmarketcap.com/exchanges/binance/'
        response = requests.get(url)
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        market_data_html_class = 'sc-936354b2-3 kcjenP cmc-table'
        table = soup.find(class_=market_data_html_class)
        column_index = 2
        trading_pairs = []

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > column_index:
                trading_pairs.append(cells[column_index].text)
                
        return trading_pairs
    
    def fetch_ohlcv(self, trading_pair, interval):
        from_ts = self.binance.parse8601(self.start_ts)
        ohlcv = self.binance.fetch_ohlcv(trading_pair, interval, since=from_ts, limit=1000)     
           
        while True:
            from_ts = ohlcv[-1][0] + 1
            new_ohlcv = self.binance.fetch_ohlcv(trading_pair, interval, since=from_ts, limit=1000)
            ohlcv.extend(new_ohlcv)
            if len(new_ohlcv) != 1000:
                break

        data_dict = {trading_pair: ohlcv}
        return data_dict
    
    def fetch_order_book(self, pair):
        url = f'https://api.binance.com/api/v3/depth?symbol={pair}&limit=10'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            time.sleep(0.5)
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"Error {response.status_code}: Too many requests. Waiting for 60 seconds...")
                time.sleep(60)
                return fetch_order_book(pair)
            else:
                print(f"HTTP error occurred: {e}")
                return {'bids': [], 'asks': []}
            
    def collect_order_books(self, cleaned_trading_pairs):
        all_data = []
        times = [self.start_time + i * timedelta(minutes=self.interval_minutes) for i in range((self.end_time - self.start_time) // timedelta(minutes=self.interval_minutes) + 1)]
        total_requests = len(cleaned_trading_pairs) * len(times)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for pair in cleaned_trading_pairs:
                for timestamp in times:
                    futures.append(executor.submit(self.fetch_order_book, pair))

            for future in tqdm(concurrent.futures.as_completed(futures), total=total_requests, unit='requests', desc='Fetching Order Book Data'):
                order_book = future.result()
                pair_index = futures.index(future) // len(times)
                timestamp_index = futures.index(future) % len(times)
                pair = self.trading_pairs[pair_index]
                timestamp = times[timestamp_index].strftime('%Y-%m-%d %H:%M:%S')

                all_data.append({
                    'Timestamp': timestamp,
                    'Trading Pair': pair,
                    'Bids': order_book['bids'],
                    'Asks': order_book['asks'],
                })

        return all_data
    
    def preprocess_order_book(self, order_book_df):
        expanded_data = []
        for index, row in order_book_df.iterrows():
            timestamp = row['Timestamp']
            pair = row['Trading Pair']      
            bids = row['Bids']
            asks = row['Asks']
            max_length = len(bids)
            
            for i in range(max_length):
                bid_price = bids[i][0]
                bid_quantity = bids[i][1]
                ask_price = asks[i][0]
                ask_quantity = asks[i][1]
                
                expanded_data.append({
                    'Timestamp': timestamp,
                    'Trading Pair': pair,
                    'Bid Price': bid_price,
                    'Bid Quantity': bid_quantity,
                    'Ask Price': ask_price,
                    'Ask Quantity': ask_quantity,
                })
                
        order_book_df = pd.DataFrame(expanded_data)
        order_book_df['Bid Price'] = pd.to_numeric(order_book_df['Bid Price'], errors='coerce')
        order_book_df['Bid Quantity'] = pd.to_numeric(order_book_df['Bid Quantity'], errors='coerce')
        order_book_df['Ask Price'] = pd.to_numeric(order_book_df['Ask Price'], errors='coerce')
        order_book_df['Ask Quantity'] = pd.to_numeric(order_book_df['Ask Quantity'], errors='coerce')

        agg_order_book = order_book_df.groupby(['Timestamp', 'Trading Pair']).agg({
            'Bid Price': ['mean', 'max', 'min'],
            'Bid Quantity': ['mean', 'sum'],
            'Ask Price': ['mean', 'max', 'min'],
            'Ask Quantity': ['mean', 'sum']
        })

        agg_order_book.columns = ['_'.join(col).strip() for col in agg_order_book.columns]

        agg_order_book.reset_index(inplace=True)
        order_book_df = agg_order_book.copy()
        del agg_order_book
        order_book_df['Timestamp'] = pd.to_datetime(order_book_df['Timestamp'])
        
        order_book_df['Bid Price Mean Change'] = order_book_df['Bid Price_mean'].pct_change()
        order_book_df['Bid Quantity Mean Change'] = order_book_df['Bid Quantity_mean'].pct_change()
        order_book_df['Ask Price Mean Change'] = order_book_df['Ask Price_mean'].pct_change()
        order_book_df['Ask Quantity Mean Change'] = order_book_df['Ask Quantity_mean'].pct_change()

        def calculate_features(group):
            group['Bid Price Mean Change'] = group['Bid Price_mean'].pct_change()
            group['Bid Quantity Mean Change'] = group['Bid Quantity_mean'].pct_change()
            group['Ask Price Mean Change'] = group['Ask Price_mean'].pct_change()
            group['Ask Quantity Mean Change'] = group['Ask Quantity_mean'].pct_change()

            group['Bid Price Rolling Mean'] = group['Bid Price_mean'].rolling(window=5).mean()
            group['Ask Price Rolling Mean'] = group['Ask Price_mean'].rolling(window=5).mean()
            group['Bid Quantity Rolling Sum'] = group['Bid Quantity_sum'].rolling(window=5).sum()
            group['Ask Quantity Rolling Sum'] = group['Ask Quantity_sum'].rolling(window=5).sum()

            group = group.fillna(0)
            return group

        order_book_df = order_book_df.groupby('Trading Pair').apply(calculate_features).reset_index(drop=True)
        numerical_columns = order_book_df.select_dtypes(include=['float64', 'int64']).columns.difference(['Timestamp'])
        scaler = StandardScaler()
        order_book_df[numerical_columns] = scaler.fit_transform(order_book_df[numerical_columns])

        return order_book_df 
                
    
    def runner(self):
        full_ohlcv = {}
        interval = '30m'
        for trading_pair in tqdm(self.trading_pairs, desc='Fetching OHLCV Data', unit='Trading Pair'):
            ohlcv = self.fetch_ohlcv(trading_pair, interval)
            full_ohlcv.update(ohlcv)
            
        data_for_ohlcv = []

        for trading_pair, ohlcv in full_ohlcv.items():
            for entry in ohlcv:
                data_for_ohlcv.append([entry[0], trading_pair] + entry[1:])
                
        ohlcv_df = pd.DataFrame(data_for_ohlcv, columns=['Timestamp', 'Trading Pair', 'Open', 'High', 'Low', 'Close', 'Volume'])
        ohlcv_df['Timestamp'] = pd.to_datetime(ohlcv_df['Timestamp'], unit='ms')
        ohlcv_df.set_index('Timestamp', inplace=True)
        ohlcv_df['24-Hour Volume'] = ohlcv_df['Volume'].rolling(window=48).sum()
        ohlcv_df['24-Hour Volume'] = ohlcv_df['24-Hour Volume'].fillna(0)
        
        trading_pairs_cleaned = [pair.replace('/', '') for pair in self.trading_pairs]
        
        data = self.collect_order_books(trading_pairs_cleaned)
        order_book_df = pd.DataFrame(data)
        order_book_df = self.preprocess_order_book(order_book_df)
        
        return ohlcv_df, order_book_df
    