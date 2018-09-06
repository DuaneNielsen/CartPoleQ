import mentality as mental
import logging

if __name__ == '__main__':

    jc = mental.Config()
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
            reloads = metadata['reloads'] if 'reloads' in metadata else 0
            log.debug('{} {} improved by {}'.format(guid, metadata['classname'], improvement))
            if reloads < 2 and improvement > most_improved:
                selected_model = metadata
                most_improved = improvement
            elif improvement > most_improved:
                log.debug('{} improved by {} but was burned after {} reloads'.format(guid, improvement, reloads))

    if selected_model is None:
        log.info('all checkpoints burned')
        exit(10)

    """ Load model from disk and flag it as reloaded """
    reloads = selected_model['reloads'] if 'reloads' in selected_model else 0
    log.info('most improved was {} {} which improved by {} and has {} reloads'.format(selected_model['guid'], \
                                                        selected_model['classname'], most_improved, reloads))
    filename = selected_model['filename']
    log.debug('loading {} model {}'.format(filename, selected_model['classname']))
    model = mental.Storeable.load(filename, jc.DATA_PATH)
    metadata = dict(model.metadata)
    if 'reloads' in model.metadata:
        metadata['reloads'] += 1
    else:
        metadata['reloads'] = 1
    mental.Storeable.update_metadata(filename, metadata, jc.DATA_PATH)

    """ train it for 5 epochs"""
    most_improved = mental.train.OneShotLoader(model)
    mental.train.run(most_improved, 'spaceinvaders/images/dev', 5)

