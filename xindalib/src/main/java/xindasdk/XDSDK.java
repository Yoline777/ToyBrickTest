package xindasdk;

public class XDSDK {
    static {
        System.loadLibrary("rkssd4j");
    }


    /*
     *  params:
     *       inputSize: 输入图像大小
     *       channel： 图像通道
     *       numClasses: SSD分类数
     *       modelPath: 模型路径
     * */
    public XDSDK(int inputSize, int channel, int numClasses, String modelPath) {

        init(inputSize, channel, numClasses, modelPath);
    }

    public void deinit() {
        native_deinit();
    }


    public static  int create_direct_texture(int texWidth, int texHeight, int format) {
        return native_create_direct_texture(texWidth, texHeight, format);
    }

    public static boolean delete_direct_texture(int texId) {
        return native_delete_direct_texture(texId);
    }

    private native int init(int inputSize, int channel, int numClasses, String modelPath);
    private native void native_deinit();

    public native int native_run(byte[] inData, float[] outputClasses);

    /*
     *  descption:
     *       检测, 只适用于Android平台
     *  params:
     *       textureId:      输入图像纹理Id
     *       outputLocations:    用于保存预测框位置(xmin, ymin, xmax, ymax)(需要后处理，具体参考PostProcess.java)
     *       outputClasses:  用于保存confidence, confidence还需要做expit处理((float) (1. / (1. + Math.exp(-x)));)
     * */
    public native int native_run(int textureId, float[] outputClasses);

    public native float[] geteye(byte[] bitmap, int width, int height, int channels, long peer);

    private static native int native_create_direct_texture(int texWidth, int texHeight, int format);

    private static native boolean native_delete_direct_texture(int texId);
}
