'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

class Local_Logger(object):

	def __init__(self, output_directory):
		self.summary_writer = SummaryWriter(log_dir=output_directory)

	def flush(self):
		self.summary_writer.flush()

	# collects data from all procs and logs the average value
	# None can be passed to exclude a process from the calculation
	def log_mean_scalar(self, key, value, x):
		self.summary_writer.add_scalar(key, float(value), x)

	# logs values sequentially
	def log_scalar_series(self, key, value, x, offset):

		if value is not None:

			# accepts a list of results if not None
			assert isinstance(value, list), type(value)

			# distribute steps over values so there is a 1-1 correspondence
			if len(value)>1:
				rem = x%len(value)
				x = [x//len(value) for _ in range(len(value))]
				x[-1] += rem
			else:
				x = [x]

		for i in range(len(value)):
			offset += x[i]
			self.summary_writer.add_scalar(key, float(value[i]), offset)