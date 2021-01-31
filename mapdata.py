import os
import csv

if __name__ == '__main__':
	with open('datamaps/datamap.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['fname', 'class', 'label'])
		PATH = os.getcwd()
		datapath = PATH + '/data'
		class_list = os.listdir(datapath)
		fnum = 1
		for i in range(len(class_list)):
			filepath = datapath+'/'+class_list[i]
			for f in os.listdir(datapath+'/'+class_list[i]):
				os.rename(filepath+'/'+f, filepath+'/'+'file'+str(fnum)+'.wav')
				writer.writerow(['file'+str(fnum)+'.wav', class_list[i], i])
				fnum += 1
