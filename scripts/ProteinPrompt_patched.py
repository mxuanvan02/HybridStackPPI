import sys
import sklearn.ensemble
import sklearn.tree
import sklearn.neighbors
import sklearn.linear_model
from types import ModuleType

# Comprehensive monkey patch for old scikit-learn models
def patch_module(old_path, new_module):
    if old_path not in sys.modules:
        sys.modules[old_path] = new_module

patch_module('sklearn.ensemble.forest', sklearn.ensemble)
patch_module('sklearn.tree.tree', sklearn.tree)
patch_module('sklearn.neighbors.dist_metrics', sklearn.neighbors)
patch_module('sklearn.neighbors.base', sklearn.neighbors)
patch_module('sklearn.linear_model.base', sklearn.linear_model)
patch_module('sklearn.linear_model.logistic', sklearn.linear_model)
patch_module('sklearn.ensemble.weight_boosting', sklearn.ensemble)

import options
import utils
import search
import predict
from argparse import Namespace
import logging
import dataclasses
from os import path

logging.basicConfig(level=logging.INFO)


def createLogger(filename: str) -> None:
    """
    Set up a logger for the main function that also saves to a log file
    """
    # get root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatterConsole = logging.Formatter('%(levelname)-8s %(message)s')
    fh.setFormatter(formatterFile)
    # add the handlers to the logger
    logger.addHandler(fh)


def main():
    """
    Main function that runs the predictions defined by command line arguments
    """
    parser = options.createCommandlineParser()
    parser.print_usage()
    prog_args: Namespace = parser.parse_args()

    try:
        if prog_args.method == "search":

            search_opts = options.SearchOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                search_opts,
                inputFile=utils.makePathAbsolute(search_opts.inputFile),
                outputDir=utils.makePathAbsolute(search_opts.outputDir)
            )
            utils.createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "check.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{fixed_opts}")
            is_valid = search.run(fixed_opts)
            exit(0)

        elif prog_args.method == "predict":

            predict_opts = options.PredictOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predict_opts,
                proteinPairs=utils.makePathAbsolute(predict_opts.proteinPairs),
                outputDir=utils.makePathAbsolute(predict_opts.outputDir)
            )
            utils.createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "check.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{fixed_opts}")
            features = predict.run(fixed_opts)
            exit(0)

    except AttributeError as e:
        print(e)
        parser.print_usage()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
