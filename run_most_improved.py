import mentality as mental

if __name__ == '__main__':

    jc = mental.JenkinsConfig()

    mdb = mental.ModelDb(jc.DATA_PATH)
    top2 = mdb.topNLossbyModelGuid(2)
    most_improved = 0
    selected_model = ''

    """ Find the modal that improved the most on the last run"""
    for guid in top2:
        if len(top2[guid]) > 2:
            improvement = top2[guid][1][0] - top2[guid][0][0]
            if improvement > most_improved:
                selected_model = top2[guid][0][1]
                most_improved = improvement

    """ Load it from disk and train it for 5 epochs"""
    model = mental.Storeable.load(selected_model['filename'], jc.DATA_PATH)
    most_improved = mental.train.OneShotLoader(model)

    mental.train.run(most_improved, 'spaceinvaders/images/dev', 5)

