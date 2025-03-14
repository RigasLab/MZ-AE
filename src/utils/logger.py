import logging

logging.basicConfig(
    filename='metrics.log',  # Specify the log file name
    level=logging.INFO,      # Set the logging level (INFO in this case)
    format='%(message)s'     # Set the format for log messages
)

class MetricsLogger:
    def __init__(self, filename, load_model = False):
        self.filename = filename
        self.headers_logged = False
        self.load_model = load_model
        self.max_len = 0

    def log_metrics(self, metrics):
        # Determine the maximum length of the metric names for alignment (only once)
        # if not self.headers_logged or self.load_model:
        self.max_len = max(len(metric) for metric in metrics.keys())

        if not self.load_model and not self.headers_logged:
            # Create and log the headers
            headers = " | ".join([f"{metric.ljust(self.max_len)}" for metric in metrics.keys()])
            with open(self.filename, 'w') as f:
                f.write(headers + "\n")

            self.headers_logged = True

        # Create and log the metric values
        values = " | ".join([f"{str(value).ljust(self.max_len)}" for value in metrics.values()])
        with open(self.filename, 'a') as f:
            f.write(values + "\n")
    