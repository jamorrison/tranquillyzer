import logging
import os

logger = logging.getLogger(__name__)


def _split_field(field):
    return [token.strip() for token in str(field).split(",") if token.strip()]


def trained_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "..", "utils")
    utils_dir = os.path.abspath(utils_dir)

    try:
        # Check if the directory exists
        if not os.path.isdir(models_dir):
            print(f"The directory '{models_dir}' does not exist.")
            return

        print("\n~~~~~~~~~~~~~~~~ CURRENTLY AVAILABLE TRAINED MODELS ~~~~~~~~~~~~~~~~")
        print(
            "\n".join(
                [
                    "-- Sequence Key:",
                    "\tNX ==> unknown sequence of length X",
                    "\tNN ==> unknown sequence of unknown length",
                    "\tA  ==> sequence of A's of unknown length",
                    "\tT  ==> sequence of T's of unknown length",
                    "",  # adds in an extra new line between key and models
                ]
            )
        )

        # Iterate over all files in the directory
        for file_name in os.listdir(models_dir):
            # Check if the file has a .h5 extension
            if file_name.endswith(".h5"):
                try:
                    seq_order, sequences, barcodes, UMIs, orientation = seq_orders(
                        os.path.join(utils_dir, "seq_orders.tsv"), file_name[:-3]
                    )

                    # Find longest seq_order name
                    longest = max([len(x) for x in seq_order])

                    # Build up elements to be printed
                    print_elements = [f"-- {file_name[:-3]}", "\tlayout (top to bottom) ==> sequence"]

                    for i in range(len(seq_order)):
                        print_elements.append(f"\t{seq_order[i]:<{longest}} ==> {sequences[i]}")

                    print_elements.append("")  # adds in an extra new line between models

                    print("\n".join(print_elements))
                except Exception:
                    print(
                        f"-- {file_name[:-3]}\n\t==> model exists in models/ directory but is undefined in utils/seq_orders.tsv\n"
                    )

    except Exception as e:
        print(f"An error occurred: {e}")


def seq_orders(file_path, model):
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"The file '{file_path}' does not exist.")
            return

        sequence_order = []
        sequences = []
        barcodes = []
        UMIs = []
        strand = ""

        # Open the file and read lines
        with open(file_path, "r") as file:
            for line in file:
                # Split the line by tabs, removing extra quote characters at the same time
                fields = line.strip().replace("'", "").replace('"', "").split("\t")

                # Check if desired model has been found
                # If so, process rest of the line
                model_name = fields[0].strip()
                if model_name == model:
                    sequence_order = _split_field(fields[1].strip())
                    sequences = _split_field(fields[2].strip())
                    barcodes = _split_field(fields[3].strip())
                    UMIs = _split_field(fields[4].strip())
                    strand = fields[5].strip()

                    return sequence_order, sequences, barcodes, UMIs, strand

                # Model name not found on this line, moving to the next one

        # If we make it here, requested model was not found
        # Verify just to be sure though
        if len(sequence_order) == 0:
            # TODO: A more well-rounded error handling set up needs to be developed
            #       This gets the job done as trying to unpack None into a tuple causes an error,
            #       but this isn't an ideal long term solution
            raise Exception(f"Requested model ({model}) not found")

    # Because we're catching the exception we just raised, this message will always print, even if we're
    # "handling" things downstream. This really needs an error handling system set up
    except Exception as e:
        print(f"An error occurred: {e}")
