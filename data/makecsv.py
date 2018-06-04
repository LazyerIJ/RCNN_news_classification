import os
import json
import csv
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",help="json list directory")
    args = parser.parse_args()
    data_path = args.data_dir

    os.chdir(data_path) 
    categories = os.listdir()
    
    print(categories)
    for category in categories:

        print('category [%s] start'%category)

        f_name = 'my_'+category+'.csv'
        json_list = os.listdir(category)

        write_csv_nums = 0

        with open(f_name,'w') as incsv:

            writer=csv.writer(incsv)
            writer.writerow(['text','type'])

            for f in json_list:
                with open(os.path.join(category,f),'r') as jf:
                    try:
                        data = json.load(jf)
                        writer.writerow([data['text'],category])
                        write_csv_nums +=1
                    except:
                        pass
        
        print('category [%s] finished'%category)
        print('all :' , len(json_list))
        print('suc :' , write_csv_nums)
    print('finished')


