import json
import csv 

# Reference the JSON file that is created using the API
filename = 'trending.json'
f = open(filename)
tiktok_data = json.load(f)

# Writing into CSV
with open('tiktok-trending.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TikTok ID", "User Verified", "Music ID", "Music Title", "Music Author", "Music URL", "Video Duration", "Play Count" ])
    for tiktok_object in tiktok_data['collector']: 
        writer.writerow([tiktok_object['id'], tiktok_object['authorMeta']['verified'], tiktok_object['musicMeta']['musicId'], tiktok_object['musicMeta']['musicName'], tiktok_object['musicMeta']['musicAuthor'], tiktok_object['musicMeta']['playUrl'], tiktok_object['videoMeta']['duration'], tiktok_object['playCount']])
f.close()