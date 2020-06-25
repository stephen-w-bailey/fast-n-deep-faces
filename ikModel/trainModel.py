import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import yaml
from pathlib import Path
import sys
import os

basePath = (Path(__file__).parent / '..').resolve()
sys.path.append(os.path.join(basePath, 'cnnModel'))
sys.path.append(os.path.join(basePath, '.'))

import ikDataLoader
import ikModel
import ikOpt


def main():
    parser = argparse.ArgumentParser(description='Train ik model through CNN approximation')
    parser.add_argument('--configFile', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    with open(args.configFile) as file:
        config = yaml.load(file)

    if 'fully_random' in config['data_params']:
        fullyRandom = config['data_params']['fully_random']
    else:
        fullyRandom = False
    if 'noise' in config['data_params']:
        noise = config['data_params']['noise']
    else:
        noise = None

    dataset = ikDataLoader.DataLoader(os.path.join(basePath,config['data_params']['sample_file']),os.path.join(basePath,config['data_params']['point_file']),fullyRandom,noise)
    data = dataset.createDataset(config['training_params']['batch_size'])

    model = ikModel.buildModel(data,config)
    loss,pointsNN,mesh = ikOpt.buildLoss(data,model,config)
    optOps = ikOpt.buildOpt(loss,config)
    ikOpt.buildSummaries(data,model,loss)

    approxVars = [v for v in tf.trainable_variables() if 'ik-model' not in v.name]
    approxSaver = tf.train.Saver(var_list=approxVars)

    lrs = [float(lr) for lr in config['training_params']['lr']]
    steps = [int(s) for s in config['training_params']['steps']]
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint,
                                           save_checkpoint_steps=50000,
                                           save_summaries_steps=100,
                                           save_summaries_secs=None) as sess:
        fileName = tf.train.latest_checkpoint(config['training_params']['approximation_checkpoint'])
        print('Loading from checkpoint '+str(fileName))
        approxSaver.restore(sess,fileName)
        try:
            print('Beginning model training')
            for lr,step in zip(lrs,steps):
                print('Setting learning rate to '+str(lr))
                sess.run(optOps['lrAssign'],feed_dict={optOps['lrPH']:lr})
                print('Setting learning rate to '+str(lr))
                for _ in tqdm.trange(step):
                    sess.run(optOps['opt'])
        except KeyboardInterrupt:
            print('Stopping early due to keyboard interrupt')

        model,data,pointsNN = sess.run((model,data,pointsNN))

if __name__=='__main__':
    main()
