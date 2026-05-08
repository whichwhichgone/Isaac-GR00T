import json

N = 6
input_json_path = "/liujinxin/liyifan/Isaac-GR00T/data.json"
output_json_path = f"/liujinxin/liyifan/Isaac-GR00T/data_{N}x_sampling.json"

with open(input_json_path, "r") as f:
    data = json.load(f)

samping_data = data[::N]

with open(output_json_path, "w") as f:
    json.dump(samping_data, f, ensure_ascii=False, indent=4)