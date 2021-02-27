import DWX_HISTORY_IO_v2_0_1_RC8 as DWX
import pandas as pd

class GENERATE_RAWDATA():
    def __init__(self, rawDataType):
        if rawDataType == "day":
            self.orgin_file_path = "/Users/lixiang/Downloads/EURUSD_20210227/EURUSD1440.hst"
            self.symbol='EURUSD'
            self.timeframe = '1440'
            self.csv_filename = 'EURUSD_day.csv'
            self.start_time = '2020-07-01 00:00:00'

        elif rawDataType == "minute_1":
            self.orgin_file_path = "/Users/lixiang/Downloads/EURUSD_20210227/EURUSD1.hst"
            self.symbol='EURUSD'
            self.timeframe = '1'
            self.csv_filename = 'EURUSD_1minute.csv'
            self.start_time = '2020-07-01 00:00:00'

        elif rawDataType == "minute_5":
            self.orgin_file_path = "/Users/lixiang/Downloads/EURUSD_20210227/EURUSD5.hst"
            self.symbol='EURUSD'
            self.timeframe = '5'
            self.csv_filename = 'EURUSD_5minute.csv'
            self.start_time = '2020-07-01 00:00:00'

        elif rawDataType == "minute_15":
            self.orgin_file_path = "/Users/lixiang/Downloads/EURUSD_20210227/EURUSD15.hst"
            self.symbol='EURUSD'
            self.timeframe = '15'
            self.csv_filename = 'EURUSD_15minute.csv'
            self.start_time = '2020-07-01 00:00:00'

    def generate_csv(self):
        
        _df = DWX.DWX_MT_HISTORY_IO(self.orgin_file_path,self.symbol,self.timeframe, self.start_time)
        _df = _df.run()
        _df.to_csv(self.csv_filename)

#day_data = GENERATE_RAWDATA("day")
#day_data.generate_csv()

#minute_data = GENERATE_RAWDATA("minute_1")
#minute_data.generate_csv()

#minute5_data = GENERATE_RAWDATA("minute_5")
#minute5_data.generate_csv()

minute15_data = GENERATE_RAWDATA("minute_15")
minute15_data.generate_csv()