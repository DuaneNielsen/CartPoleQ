from mentality import JenkinsConfig, ModelDb


jc =JenkinsConfig()
mdb = ModelDb(jc.DATA_PATH)
mdb.sync_to_elastic() 