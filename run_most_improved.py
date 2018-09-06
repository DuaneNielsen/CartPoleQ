import mentality as mental
import logging

if __name__ == '__main__':

    jc = mental.JenkinsConfig()
    logfile = jc.getLogPath('most_improved.log')
    logging.basicConfig(filename=logfile.absolute())
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    mdb = mental.ModelDb(jc.DATA_PATH)
    top2 = mdb.topNLossbyModelGuid(2)
    most_improved = 0
    selected_model = None

    """ Find the modal that improved the most on the last run"""
    for guid in top2:
        if len(top2[guid]) == 2:
            improvement = top2[guid][1][0] - top2[guid][0][0]
            metadata = top2[guid][0][1]
            reloaded = metadata['reloaded'] if 'reloaded' in metadata else 0
            log.debug('{} improved by {}'.format(guid, improvement))
            if reloaded < 3 and improvement > most_improved:
                selected_model = metadata
                most_improved = improvement
            elif improvement > most_improved:
                log.debug('{} improved by {} but was burned after {} reloads'.format(guid, improvement, reloaded))

    if selected_model is None:
        log.info('all checkpoints burned')
        exit(10)

    """ Load model from disk and flag it as reloaded """
    log.info('most improved was {} which improved by {}'.format(selected_model['guid'], most_improved))
    filename = selected_model['filename']
    model = mental.Storeable.load(filename, jc.DATA_PATH)
    metadata = dict(model.metadata)
    if 'reloaded' in model.metadata:
        metadata['reloaded'] += 1
    else:
        metadata['reloaded'] = 1
    mental.Storeable.update_metadata(filename, metadata, jc.DATA_PATH)

    """ train it for 5 epochs"""
    most_improved = mental.train.OneShotLoader(model)
    mental.train.run(most_improved, 'spaceinvaders/images/dev', 5)

