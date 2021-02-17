import java.io.*;
import java.nio.file.*;
import java.util.*;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.semanticweb.owlapi.model.parameters.*;
import org.semanticweb.elk.reasoner.config.*;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner;
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration;

import py4j.GatewayServer;


public class Main {

    Logger logger;

    @Parameter(names={"--port", "-p"}, required=false)
    int port = 25333;

    @Parameter(names={"--connection-timeout", "-ct"}, required=false)
    int connectionTimeout = 0;

    @Parameter(names={"--read-timeout", "-rt"}, required=false)
    int readTimeout = 0;

    public Main() {
	logger = LoggerFactory.getLogger(Main.class);
    }

    public void run() throws Exception{

	logger.info("Run function is excecuted");
	
	GatewayServer server = new GatewayServer(
            this, this.port, this.connectionTimeout, this.readTimeout);
        server.start();
    }

    public static void main(String[] args) {
	Main main = new Main();
	JCommander jcom = JCommander.newBuilder()
            .addObject(main)
            .build();
	try {
	    jcom.parse(args);
	    main.run();
	} catch (Exception e) {
	    e.printStackTrace();
	    jcom.usage();
	}
    }
}
