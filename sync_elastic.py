from mentality import Config, ModelDb


jc =Config()
mdb = ModelDb(jc.DATA_PATH)
mdb.sync_to_elastic()