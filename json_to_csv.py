import json
import csv

def read_json_results(surrogate):
    json_file = "LR_{}.json".format(surrogate)
    with open(json_file) as f:
        return json.load(f)
    

def write_to_csv(results, surrogate):
    csv_file = "LCP_{}.csv".format(surrogate)
    max_epoch = 15
    if surrogate == "data20" or surrogate == "data30":
        max_epoch = 50
    elif surrogate == "data207":
        max_epoch = 100

    with open(csv_file, 'w') as f:
        fieldnames = ["index", "checkpoint", "best_acc", 
                      "con_poster_prob",
                      #"con_poster_mean_prob", 
                      #"opt_poster_mean_prob", 
                      #"opt_poster_prob",
                       "est_time"
                       ]
        for e in range(max_epoch):
            fieldnames.append("epoch_{}".format(e + 1))
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        indices = []
        for key in sorted(results.keys()):
            indices.append(int(key))

        for index in sorted(indices):
            try:                
                r = results[str(index)]
                lr = r["lr"]
                row = {}
                row['index'] = index
                row['checkpoint'] = r["checkpoint"]
                row['best_acc'] = r["max_acc"]
                if "conservative-posterior_prob_x_greater_than" in r:
                    con_prob_result = r["conservative-posterior_prob_x_greater_than"]
                    if "est_time" in con_prob_result:
                        row['est_time'] = con_prob_result["est_time"]
                    row['con_poster_prob'] = con_prob_result["y_predict"]

                #row['con_poster_mean_prob'] = r["conservative-posterior_mean_prob_x_greater_than"]["y_predict"]
                #row['opt_poster_mean_prob'] = r["optimistic-posterior_mean_prob_x_greater_than"]["y_predict"]
                #row['opt_poster_prob'] = r["optimistic-posterior_prob_x_greater_than"]["y_predict"]
                for i in range(max_epoch):
                    row["epoch_{}".format(i+1)] = lr[i]

                writer.writerow(row)

            except Exception as ex:
                print("Result {} is skipped due to exception.".format(index))
                index += 1
                continue        
 

if __name__ == "__main__":
    s = "data20"
    r = read_json_results(s)
    write_to_csv(r, s)