from mowl.model import Model



class WalkRdfOwl(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.dataset.url = 'http://aber-owl.net/rdfwalker/t.owl'
        
    def gen_graph(input, output, mapping, undirected, format, ontology_directory):

        withMap = False

 #       tmp_data_file = java.io.File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')

#        ont_manager = OWLManager.createOWLOntologyManager()

        # ontology_set_files = os.listdir(ontology_directory)
        # ont_hash_set = java.util.LinkedHashSet()
        # for file in ontology_set_files:
        #     ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+ontology_directory+file)))
        # ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+input)))

        # ontology = ont_manager.createOntology(IRI.create("http://aber-owl.net/rdfwalker/t.owl"), ont_hash_set)

        # factory =  ont_manager.getOWLDataFactory()

        # progressMonitor = ConsoleProgressMonitor()
        # config = SimpleConfiguration(progressMonitor)
        
        # reasoner_factory = ElkReasonerFactory()
        # reasoner = reasoner_factory.createReasoner(ontology,config)

#
#        inferred_axioms = InferredClassAssertionAxiomGenerator().createAxioms(factory, reasoner)

        self.dataset.infer_axioms()
 
        # counter = 0
        # for axiom in inferred_axioms:
        #     ont_manager.addAxiom(ontology, axiom)
        #     counter += 1
            
        # ont_manager.saveOntology(ontology, IRI.create(tmp_data_file.toURI()))


        print(str(counter) + " axioms inferred.")

        
        filename = tmp_data_file.toURI()
        model = ModelFactory.createDefaultModel()
        infile = FileManager.get().open(filename.toString())

        model.read(infile, None, format)

        counter = 1
        dict = {} # maps IRIs to ints; for input to deepwalk

        out_file = open(output, 'w')
        for stmt in model.listStatements():
            pred = stmt.getPredicate()
            subj = stmt.getSubject()
            obj = stmt.getObject()

            
            if withMap:

                if (subj.isURIResource() and obj.isURIResource()):
                
                    if not pred in dict:
                        dict[pred] = counter
                        counter += 1
                    
                    if not subj in dict:
                        dict[subj] = counter
                        counter += 1
                
                    if not obj in dict:
                        dict[obj] = counter
                        counter += 1
                    
                    predid = dict[pred]
                    subjid = dict[subj]
                    objid = dict[obj]
                    
                    # generate three nodes and directed edges
                    out_file.write(str(subjid)+"\t"+str(objid)+"\t"+str(predid)+"\n")
                    
                    # add reverse edges for undirected graph; need to double the walk length!
                    if (undirected):
                        out_file.write(str(objid)+"\t"+str(subjid)+"\t"+str(predid)+"\n")

            else:
                if (subj.isURIResource() and obj.isURIResource()):
                    
                    # generate three nodes and directed edges
                    out_file.write(subj+"\t"+obj+"\t"+pred+"\n")
                    
                    # add reverse edges for undirected graph; need to double the walk length!
                    if (undirected):
                        out_file.write(obj+"\t"+subj+"\t"+pred+"\n")                        

        map_file = open(mapping, 'w')

        if (withMap):
            for key, val in dict.items():
                map_file.write(str(key)+'\t'+str(val)+'\n')