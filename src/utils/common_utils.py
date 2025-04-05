import inspect
import logging
import os
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue
from typing import Optional


class AppLog:
	_instance: Optional['AppLog'] = None
	_logger: Optional[logging.Logger] = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(AppLog, cls).__new__(cls)
			cls._instance._initialize_logger()
		return cls._instance

	def _initialize_logger(self) -> None:
		"""Initialize the logger with rotating file handler"""
		if self._logger is None:
			self._logger = logging.getLogger('ApplicationLogger')
			self._logger.setLevel(logging.INFO)  # Default level

			log_que = Queue(maxsize=1024)
			q_handle = QueueHandler(log_que)
			file_dir = Path.cwd(__file__).parent.resolve()
			logdir = file_dir.parent.parent / 'log'
			logfile = logdir / 'application.log'
			print('Logging to {}'.format(logfile))

			handler = RotatingFileHandler(logfile,
										  maxBytes=3 * 1024 * 1024,  # 3MB
										  backupCount=5, encoding='utf-8')

			# Format for the log messages
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)

			# Console handler
			# console_handler = logging.StreamHandler(sys.stdout)
			# console_handler.setFormatter(formatter)

			# self._logger.addHandler(console_handler)
			self._q_listener = QueueListener(log_que, handler)
			self._logger.addHandler(q_handle)
			self._q_listener.start()

	@classmethod
	def set_level(cls, level: str) -> None:
		"""Set the logging level"""
		if cls._instance is None:
			cls._instance = cls()

		level_map = {'DEBUG'   : logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING,
					 'ERROR'   : logging.ERROR,
					 'CRITICAL': logging.CRITICAL}

		cls._instance._logger.setLevel(level_map.get(level.upper(), logging.INFO))

	@classmethod
	def shut_down(cls) -> None:
		"""Shuts down the logger"""
		if cls._instance is not None:
			try:
				cls._instance._q_listener.stop()
			except AttributeError:
				pass

	def _log(self, level: str, message: str) -> None:
		"""Internal method to handle logging with caller information"""
		# Get caller frame information
		caller_frame = inspect.currentframe().f_back.f_back
		filename = os.path.basename(caller_frame.f_code.co_filename)
		func_name = caller_frame.f_code.co_name

		# Combine filename, function name and message
		full_message = f"[{filename}:{func_name}] {message}"

		if level == 'DEBUG':
			self._logger.debug(full_message)
		elif level == 'INFO':
			self._logger.info(full_message)
		elif level == 'WARNING':
			self._logger.warning(full_message)
		elif level == 'ERROR':
			self._logger.error(full_message)
		elif level == 'CRITICAL':
			self._logger.critical(full_message)

	@classmethod
	def debug(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('DEBUG', message)

	@classmethod
	def info(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('INFO', message)

	@classmethod
	def warning(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('WARNING', message)

	@classmethod
	def error(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('ERROR', message)

	@classmethod
	def critical(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('CRITICAL', message)
