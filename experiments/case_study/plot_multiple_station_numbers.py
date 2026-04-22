import os
import csv
import matplotlib.pyplot as plt


MFCS_RGB = [(153, 153, 153), (0, 101, 189), (0, 0, 0), (159, 186, 54)]


def set_cols():
    for i, sett in enumerate(MFCS_RGB):
        temp_list = []
        for elem in sett:
            temp_list.append(elem / 255)
        MFCS_RGB[i] = (temp_list[0], temp_list[1], temp_list[2])
    return MFCS_RGB


def plot_variable_num_stations(result_dict, res_dir):
    plt.figure()
    num_stations_vals = [float(x) for x in list(result_dict.keys())]
    max_phase_vals = [float(y) for y in list(result_dict.values())]
    plt.plot(
        num_stations_vals,
        max_phase_vals,
        ".-",
        color=MFCS_RGB[2],
        markersize=8,
        linewidth=1,
    )
    plt.xticks(num_stations_vals)
    plt.xlabel("Number of stations", fontname="Liberation Serif", fontsize=11)
    plt.ylabel(
        "Max phase welding lenght [mm]", fontname="Liberation Serif", fontsize=11
    )
    plt.title(
        "Time balancing for variable station numbers",
        fontname="Liberation Serif",
        fontsize=13,
    )
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    plt.savefig(os.path.join(res_dir, "plot_multiple_station_numbers.svg"), dpi=1200)
    plt.close()


def get_csv_data(csv_file):
    res = {}
    with open(csv_file, mode="r", encoding="utf-8") as f:
        csv_r = csv.reader(f)
        next(csv_r)
        for row_val in csv_r:
            res[row_val[0]] = row_val[1]
    return res


if __name__ == "__main__":
    CSV_FNAME = "experiments/case_study/test_multiple_station_numbers.csv"
    set_cols()
    multiple_stations_res = get_csv_data(CSV_FNAME)
    print(multiple_stations_res)

    plot_variable_num_stations(multiple_stations_res, os.path.dirname(CSV_FNAME))
