import json
from datetime import datetime
from collections import defaultdict
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hourly_sentiments = defaultdict(float)
daily_sentiments = defaultdict(float)
hourly_tweet_counts = defaultdict(int)
daily_tweet_counts = defaultdict(int)


def get_sentiment(sentiment_data):
    if isinstance(sentiment_data, (int, float)):
        return sentiment_data
    elif isinstance(sentiment_data, dict) and 'score' in sentiment_data:
        return float(sentiment_data['score'])
    return 0.0


def get_date_hour(timestamp):
    timestamp_object = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    date = timestamp_object.strftime('%Y-%m-%d')
    hour = timestamp_object.strftime('%H:00')
    return date, hour


def analysis_tweets(tweet_line):
    if tweet_line.strip() in ('{', '}', '[', ']', '{]}') or '"rows":[' in tweet_line:
        return
    try:
        cleaned_line = tweet_line.rstrip(",\n\r ").strip()
        if not cleaned_line.startswith('{') or cleaned_line == '{}]}':
            return
        tweet = json.loads(cleaned_line)
        tweet_data = tweet.get('doc', {}).get('data', {})
        sentiment_data = tweet_data.get('sentiment')
        created_at = tweet_data.get('created_at')

        if sentiment_data is None or created_at is None:
            return

        sentiment = get_sentiment(sentiment_data)
        date_str, hour_str = get_date_hour(created_at)

        hourly_sentiments[hour_str] += sentiment
        daily_sentiments[date_str] += sentiment
        hourly_tweet_counts[hour_str] += 1
        daily_tweet_counts[date_str] += 1
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from line: {tweet_line}")
        print(e)


if __name__ == '__main__':
    path_to_file = './resources/twitter-50mb.json'

    with open(path_to_file, 'r') as file:
        for i, line in enumerate(file):
            if i % size == rank:
                analysis_tweets(line.strip())

    all_hourly_sentiments = comm.gather(hourly_sentiments, root=0)
    all_daily_sentiments = comm.gather(daily_sentiments, root=0)
    all_hourly_tweet_counts = comm.gather(hourly_tweet_counts, root=0)
    all_daily_tweet_counts = comm.gather(daily_tweet_counts, root=0)

    if rank == 0:
        for dict_list in [all_hourly_sentiments, all_daily_sentiments, all_hourly_tweet_counts, all_daily_tweet_counts]:
            for d in dict_list[1:]:
                for k, v in d.items():
                    dict_list[0][k] += v

        happiest_hour = max(all_hourly_sentiments[0], key=all_hourly_sentiments[0].get)
        happiest_day = max(all_daily_sentiments[0], key=all_daily_sentiments[0].get)
        most_active_hour = max(all_hourly_tweet_counts[0], key=all_hourly_tweet_counts[0].get)
        most_active_day = max(all_daily_tweet_counts[0], key=all_daily_tweet_counts[0].get)

        print(f'Happiest Hour: {happiest_hour}, Sentiment: {all_hourly_sentiments[0][happiest_hour]}')
        print(f'Happiest Day: {happiest_day}, Sentiment: {all_daily_sentiments[0][happiest_day]}')
        print(f'Most Active Hour: {most_active_hour}, Tweet Count: {all_hourly_tweet_counts[0][most_active_hour]}')
        print(f'Most Active Day: {most_active_day}, Tweet Count: {all_daily_tweet_counts[0][most_active_day]}')
