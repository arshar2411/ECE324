import json
import csv 

# Reference the JSON file that is created using the API
filename = 'tiktok-sample.json'
with open(filename, 'r') as infile:
    data = infile.read()
    new_data = data.replace('}\n{', '},{')
    tiktok_data = json.loads(f'[{new_data}]')

with open('tiktok-sample.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TikTok ID", "Music ID", "Music Title", "Music URL", "Play Count" ])
    for tiktok_object in tiktok_data: 
        writer.writerow([tiktok_object['id'], tiktok_object['music']['id'], tiktok_object['music']['title'], tiktok_object['music']['playUrl'], tiktok_object['stats']['playCount']])