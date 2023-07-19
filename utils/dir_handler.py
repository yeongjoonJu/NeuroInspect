import os, json
from PIL import Image
from utils.ops import to_rgb

def __get_filenames_in_a_folder(folder: str):
    """
    returns the list of paths to all the files in a given folder
    """

    files = os.listdir(folder)
    files = [f"{folder}/" + x for x in files]
    return files


def check_if_a_different_config_exists_with_same_name(filename, data):
    overwrite_neurons = False

    config_already_exists = os.path.exists(filename)
    if config_already_exists == True:
        existing_config = json.load(open(filename))
        for (k1, v1), (k2, v2) in zip(existing_config.items(), data.items()):

            assert (
                k1 == k2
            ), f"Expected keys in config to be the same, but got {k1} and {k1}"

            if k1 != "neuron_idx":
                ## if config fully matches, then do not overwrite existing neurons
                if v1 == v2:
                    pass
                else:
                    overwrite_neurons = True
                    break
    else:
        overwrite_neurons = True

    return overwrite_neurons


def make_folder_if_it_doesnt_exist(name):

    if name[-1] == "/":
        name = name[:-1]

    folder_exists = os.path.exists(name)

    if folder_exists == True:
        num_files = len(__get_filenames_in_a_folder(folder=name))
        if num_files > 0:
            UserWarning(f"Folder: {name} already exists and has {num_files} items")
    else:
        os.mkdir(name)


def check_and_write_config(
    save_dir, experiment_name,
    neuron_idx, image_size, iters, lr, batch_size,
    weight_decay, truncation_psi=None, n_samples=None
):
    data = {
        "experiment_name": experiment_name,
        "neuron_idx": neuron_idx,
        "image_size": image_size,
        "iters": iters, "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
    }
    if n_samples is not None:
        data.update({"n_samples": n_samples})
    if truncation_psi is not None:
        data.update({"truncation_psi": truncation_psi})
        
    folder_name = save_dir + "/configs"

    make_folder_if_it_doesnt_exist(name=folder_name)
    filename = folder_name + "/" + experiment_name + ".json"

    overwrite_neurons = check_if_a_different_config_exists_with_same_name(
        filename=filename, data=data
    )

    ## if this is true then either the config does NOT already exists or exists with different params
    if overwrite_neurons == True:
        with open(filename, "w") as fp:
            json.dump(data, fp, indent=2)
        fp.close()

    return overwrite_neurons


def save_activations(acts, neuron_idx, filename_no_ext, dream_acts=None, text=None):
    save_path = filename_no_ext+".json"
    act_dict = {}
    for i in range(acts.shape[0]):
        act_dict[f"{neuron_idx}_{i}.jpg"] = {
            "activation": acts[i].item(),
        }
        if text is not None:
            if type(text) is list:
                for k in range(len(text)):
                    act_dict[f"{neuron_idx}_{i}.jpg"].update({f"text{k}": text[k]})
            else:
                act_dict[f"{neuron_idx}_{i}.jpg"].update({"text": text})
        if dream_acts is not None:
            act_dict[f"{neuron_idx}_{i}.jpg"].update({"dream_act": dream_acts[i].item()})
    
    with open(save_path, "w") as fout:
        json.dump(act_dict, fout, indent=2)


def set_experiment_dir(save_dir, experiment_name, overwrite_experiment, starting_time, folder_name="sAMS"):
    sAMS_folder = f"{save_dir}/{folder_name}"

    # creating a subfolder for sAMS (if not exists)
    folder_exists = os.path.exists(sAMS_folder)
    if folder_exists == False:
        os.mkdir(sAMS_folder)
        print(f"Subfolder for sAMS created at {sAMS_folder}")
    else:
        print(f"Using existing sAMS folder at {sAMS_folder}")

    # start generation
    experiment_name = (
        str(starting_time) if experiment_name is None else experiment_name
    )
    print(f"Experiment name: {experiment_name}")
    experiment_folder = sAMS_folder + "/" + experiment_name

    # TODO: check if experiment folder exists (well, it shouldn't, but still)
    experiment_folder_exists = os.path.exists(experiment_folder)

    if not experiment_folder_exists:
        os.mkdir(experiment_folder)
    elif experiment_folder_exists == True and overwrite_experiment == True:
        print(f"Overwriting experiment: {experiment_name}")
    else:
        raise Exception(
            f"an experiment with the name {experiment_name} already exists, set overwrite_experiment = True if you want to overwrite it"
        )

    return experiment_folder


def save_illusion_results(images, acts, dir_name, neuron_idx, \
                        dreams=None, dream_acts=None, class_name=None):
    images = to_rgb(images.cpu().detach())
    acts = acts.cpu().detach()
    images = [Image.fromarray(images[b], "RGB") for b in range(images.shape[0])]
    
    if dreams is not None:
        dreams = to_rgb(dreams.cpu().detach())
        dream_acts = dream_acts.cpu().detach()
        dreams = [Image.fromarray(dreams[b], "RGB") for b in range(dreams.shape[0])]
    
    name_no_ext = f"{dir_name}/{neuron_idx}"
    for b in range(len(images)):
        images[b].save(name_no_ext+f"_image_{b}.jpg")
        if dreams is not None:
            dreams[b].save(name_no_ext+f"_dream_{b}.jpg")

    save_activations(acts, neuron_idx, name_no_ext, dream_acts=dream_acts, text=class_name)