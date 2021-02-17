import py4j.GatewayServer;

public class Test {

    public static void main(String[] args) {
        Test app = new Test ();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();
    }
}
