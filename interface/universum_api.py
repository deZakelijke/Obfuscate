import os
import requests
import json


def post_batch(poses, bundle_loc="bundle.zip",):
    file = open(bundle_loc, 'rb')     # flat structure zip file

    files = {'images': (bundle_loc, file)}
    payload = {
        "glasses": {
            "models":["Enzo"],
            "materials":[{
                "name":"Gold Potato",
                "frame":{"color":"rgb(0, 0, 0)", "opacity":0.0},
                "glass":{"color":"rgb(0, 0, 0)", "opacity":0.0}
            }]
        },
        "poses": poses
    }


    data = {
        "variants": json.dumps(payload),
        # "email": "antons@live.nl",
        # "email": "lg.weitkamp@gmail.com",
    }

    r = requests.post(post_url, files=files, data=data)
    file.close()

    content = json.loads(r.content)
    # print("content", content)
    process_id = content["processId"]
    return process_id


def get_status(process_id):
    url = get_url.format(process_id)
    r = requests.head(url)
    print(url, r.status_code)
    if r.status_code != 200:
        return False
    return True


def get_processed(process_id, filename):
    url = get_url.format(process_id)

    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            return False
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return True

