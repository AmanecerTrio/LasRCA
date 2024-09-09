from datetime import datetime
import pytz


class TimeHandler:
    @staticmethod
    def datetime_to_timestamp(datetime_str: str) -> int:
        """
        将日期时间字符串转换为时间戳（考虑时区）.
        :param datetime_str: %Y-%m-%d %H:%M:%S格式的日期时间字符串.
        :return: int, 10位时间戳
        """
        timezone = pytz.timezone("Asia/Shanghai")
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return int(timezone.localize(dt).timestamp())

    @staticmethod
    def timestamp_to_datetime(timestamp: int):
        """
        将时间戳转换为日期时间字符串（Asia/Shanghai时区）.
        :param timestamp: 10位时间戳.
        :return: str, %Y-%m-%d %H:%M:%S格式的日期时间字符串.
        """
        timezone = pytz.timezone("Asia/Shanghai")
        dt = pytz.datetime.datetime.fromtimestamp(timestamp, timezone)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def get_date_timestamp_list(date: str) -> list:
        """
        给定一个日期, 获取当天每分钟的时间戳, 便于划分时间窗口.
        :param date: %Y-%m-%d格式的日期字符串.
        :return: list, 包含了当天每分钟对应时间戳的数组.
        """
        ...
