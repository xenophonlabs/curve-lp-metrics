"""
Train hyperparameters for BOCD models. Write them to hyperparameters.json.
"""
import warnings
import pandas as pd
import traceback
from datetime import datetime
from curvemetrics.src.classes.datahandler import DataHandler
from curvemetrics.src.classes.modelsetup import ModelSetup
from curvemetrics.src.classes.logger import Logger

# Suppress all runtime warnings - they pertain to large integers and divisions by 0 in tuning.
warnings.filterwarnings("ignore")

POOL_METRICS = {'shannonsEntropy', 'giniCoefficient', 'netSwapFlow', 'netLPFlow', '300.Markout', 'sharkflow'}
TOKEN_METRICS = {'logReturns'}

def main():
    """
    Train and test hyperparameters for each combination of pool/token and metric. 
    Save corresponding results in hyperparameters.json and plots in ./figs directory.
    """

    logger = Logger('./logs/tune.log').logger
    datahandler = DataHandler(logger=logger)
    tuner = ModelSetup(datahandler, logger=logger, plotting=True)

    try:

        logger.info(f"Tuning hyperparameters...\n")
        logger.info(f"Using margin: {tuner.margin}")
        logger.info(f"Using alpha: {tuner.alpha}")
        logger.info(f"Testing grid of length {len(tuner.GRID)}\n")

        ### POOL METRICS TRAINING

        train_pool = "0xceaf7747579696a2f0bb206a14210e3c9e6fb269" # UST Metapool
        train_start = datetime.timestamp(datetime(2022, 1, 1))
        train_end = datetime.timestamp(datetime(2022, 6, 1))

        train_params = tuner.train(train_pool, POOL_METRICS, train_start, train_end, 'pool')

        ### POOL METRICS TESTING
        
        test_end = datetime.timestamp(datetime(2023, 5, 1))
        pool_results = []

        for pool in datahandler.pool_metadata:
            if pool == train_pool:
                continue
            try:
                test_start = min(datetime.timestamp(datetime(2022, 1, 1)), datahandler.pool_metadata[pool]['creationDate'])
                results = tuner.test(pool, POOL_METRICS, test_start, test_end, train_params, 'pool')
                pool_results += results
            except Exception as e:
                logger.error(f"Failed for {datahandler.pool_metadata[pool]['name']}: {traceback.print_exc()}\n")
                continue
        
        pd.DataFrame(pool_results, columns=['pool', 'metric', 'F', 'P', 'R', 'params']).to_csv('./results/pool_results.csv', index=False)
        
        ### TOKEN METRICS TRAINING

        train_token = '0xa693b19d2931d498c5b318df961919bb4aee87a5' # UST
        train_start = datetime.timestamp(datetime(2022, 1, 1))
        train_end = datetime.timestamp(datetime(2022, 6, 1))

        train_params = tuner.train(train_token, TOKEN_METRICS, train_start, train_end, 'token')
        
        ### TOKEN METRICS TESTING

        test_start = datetime.timestamp(datetime(2022, 1, 1))
        test_end = datetime.timestamp(datetime(2023, 5, 1))
        token_results = []
            
        for token in datahandler.token_metadata:
            if token in ['0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2']:
                continue
            if token == train_token:
                continue
            try:
                results = tuner.test(token, TOKEN_METRICS, test_start, test_end, train_params, 'token')
                token_results += results
            except Exception as e:
                logger.error(f"Failed for {datahandler.token_metadata[token]['symbol']}: {traceback.print_exc()}\n")
                continue
        
        pd.DataFrame(token_results, columns=['token', 'metric', 'F', 'P', 'R', 'params']).to_csv('./results/token_results.csv', index=False)

    except Exception as e:
        logger.error(f"Failed to train and/or test bocd hyperparameters: {e}.\n")
        raise e

    finally:
        datahandler.close()
    
if __name__ == "__main__":
    main()