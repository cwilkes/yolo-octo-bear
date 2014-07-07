import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class CLI {

    private static int readNextInt(BufferedReader br) throws NumberFormatException, IOException {
        return Integer.parseInt(br.readLine());
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int N = readNextInt(br);
        int[] data = new int[N];
        for (int i = 0; i < N; i++) {
            data[i] = readNextInt(br);
        }
        CollageMaker cm = new CollageMaker();
        int[] ret = cm.compose(data);
        for (int i : ret) {
            System.out.println(i);
        }
        System.out.flush();
    }

}
