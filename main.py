from river import drift, preprocessing
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class AnomalyDetection:
    def __init__(self):
        '''
        Initialize the model and preprocessing tools.
        
        - ADWIN: A drift detection method that adapts to changes in the data distribution.
        - StandardScaler: A scaler for standardizing features.
        - IsolationForest: An outlier detection algorithm.
        '''
        self.adwin = drift.ADWIN(delta=0.002)  # Adaptive windowing for concept drift
        self.scaler = preprocessing.StandardScaler()  # Real-time standard scaler for stream data
        self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01)
        self.df = pd.DataFrame(columns=['timestamp', 'value', 'anomaly'])
        self.model_trained = False  # Flag to check if the model has been trained

    def simulate_data_stream(self, num_points=500):
        '''
        Simulate data stream with regular patterns, seasonal elements, and random noise.
        
        Args:
            num_points (int): Number of data points to simulate.
        
        Returns:
            pd.DataFrame: Simulated data stream with timestamps and values.
        '''
        np.random.seed(42)
        timestamps = pd.date_range(start='2024-01-01', periods=num_points, freq='min')
        seasonal_pattern = np.sin(np.linspace(0, 20 * np.pi, num_points))  # Seasonal element
        noise = np.random.normal(0, 0.1, num_points)  # Random noise
        trend = np.linspace(0, 1, num_points)  # Gradual trend
        values = 10 + seasonal_pattern + trend + noise
        return pd.DataFrame({'timestamp': timestamps, 'value': values})

    def process_data_stream(self, data_stream):
        '''
        Process the data stream in real-time and detect anomalies.
        
        Args:
            data_stream (pd.DataFrame): DataFrame containing the data stream.
        '''
        for index, row in data_stream.iterrows():
            timestamp = row['timestamp']
            value = row['value']

            try:
                # Update ADWIN with the new data point and handle concept drift
                if self.adwin.update(value):
                    print(f"Concept drift detected at {timestamp}")

                # Update the real-time scaler and transform the value
                self.scaler.learn_one({'value': value})  # Update the scaler
                scaled_value = self.scaler.transform_one({'value': value})['value']  # Transform the value

                # Append the processed data
                self.df = pd.concat([self.df, pd.DataFrame([{'timestamp': timestamp, 'value': value, 'anomaly': 0}])], ignore_index=True)

                # Train the model if it has not been fitted yet
                if len(self.df) > 100:  # Example threshold for batch size
                    if not self.model_trained:
                        self.model.fit(self.df[['value']])
                        self.model_trained = True

                # Make anomaly prediction only if the model is trained
                if self.model_trained:
                    anomaly_prediction = self.model.predict([[scaled_value]])
                    is_anomaly = 1 if anomaly_prediction == -1 else 0
                    self.df.at[self.df.index[-1], 'anomaly'] = is_anomaly
                else:
                    # No prediction if model is not trained
                    self.df.at[self.df.index[-1], 'anomaly'] = 0

                # Visualize in real-time after processing every 10 points
                if index % 10 == 0:
                    self.visualize_real_time(index)

            except Exception as e:
                print(f"Error processing data point at index {index}: {e}")

    def visualize_real_time(self, index):
        '''
        Real-time visualization of data and anomalies.
        
        Args:
            index (int): The current index to visualize data.
        '''
        plt.clf()
        subset = self.df.iloc[max(0, index - 100):index]  # Show last 100 points
        plt.plot(subset['timestamp'], subset['value'], color='orange', label='Data Stream')
        anomalies = subset[subset['anomaly'] == 1]
        plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomalies')
        plt.title("Real-Time Anomaly Detection")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.pause(0.01)  # Small pause to simulate real-time streaming

    def run(self):
        '''
        Main method to simulate the stream and run anomaly detection.
        '''
        plt.ion()  # Interactive mode on for real-time plotting
        data_stream = self.simulate_data_stream(num_points=500)
        self.process_data_stream(data_stream)
        plt.ioff()  # Interactive mode off
        plt.show()

if __name__ == "__main__":
    anomaly_detector = AnomalyDetection()
    anomaly_detector.run()
