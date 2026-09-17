package com.pharmaflow.pillcount

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.SharedPreferences
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.util.Log
import java.util.Collections

/**
 * The one thing Python does not do on this phone: the forward pass.
 *
 * pillsort runs the model through `cv2.dnn` on a PC, and that is not available here --
 * Chaquopy's package repository stops at OpenCV 4.5.1, whose ONNX importer predates the
 * attention blocks in `pillsort-seg.onnx` and cannot parse the file at all. ONNX Runtime
 * can, so the session lives in Kotlin and Python calls into it.
 *
 * The interface is deliberately dumb: bytes in, bytes out. Python hands over the blob
 * `cv2.dnn.blobFromImage` already produces (NCHW float32, little-endian) and gets the raw
 * output tensors back the same way, then reshapes them with [shapes]. Nothing about
 * YOLO -- anchors, mask prototypes, NMS -- is known on this side of the call, because
 * all of that is `pillsort/backend.py`'s and there is one copy of it.
 */
class OrtEngine(modelBytes: ByteArray, prefs: SharedPreferences? = null,
                stamp: String = "") {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    /** Which execution provider [session] ended up on, for the log and the app's footer. */
    val provider: String

    /** What that provider measured, in milliseconds a pass. 0 when it was not raced. */
    val providerMs: Double

    private val session: OrtSession

    init {
        // Four threads at most. A phone's big cores are four; asking for every core lets
        // ORT schedule onto the little ones too, where the per-thread sync costs more
        // than the work they contribute.
        val threads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)

        val builders = linkedMapOf<String, () -> OrtSession>(
            // NNAPI hands the graph to whatever the phone has for this: a DSP, a GPU, an
            // NPU. On a device with any of them it is a multiple faster than CPU kernels,
            // and on a device without them it is a multiple SLOWER than plain CPU, which
            // is not a thing you can tell by asking -- see the note on racing below.
            "nnapi" to {
                env.createSession(modelBytes, OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(threads)
                    addNnapi()
                })
            },
            // XNNPACK ships in the same AAR and has hand-written ARM NEON kernels for
            // exactly this export's float32 convolutions. It brings its own thread pool
            // and asks that ORT's intra-op parallelism be left at one, or the two fight
            // over the same cores.
            "xnnpack" to {
                env.createSession(modelBytes, OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(1)
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                })
            },
            "cpu" to {
                env.createSession(modelBytes, OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(threads)
                })
            },
        )

        // KEYED BY THE BUILD, and it has to be. SharedPreferences survive an app
        // update, so without the stamp a phone that raced ORT 1.17 against this model
        // would go on trusting that answer after both had been replaced by an install.
        // It is the same stamp Art.unpack uses, for the same reason.
        val key = "$KEY_PROVIDER-$stamp"
        val remembered = prefs?.getString(key, null)
        var chosen: OrtSession? = null
        var chosenName = ""
        var chosenMs = 0.0

        if (remembered != null && builders.containsKey(remembered)) {
            // Raced once already on THIS phone with THIS model. Building one session
            // instead of three takes seconds off every launch after the first.
            chosen = try {
                builders[remembered]!!()
            } catch (t: Throwable) {
                Log.w(TAG, "remembered provider $remembered no longer builds", t)
                null
            }
            if (chosen != null) {
                chosenName = remembered
                Log.i(TAG, "provider $remembered, remembered from an earlier run")
            }
        }

        if (chosen == null) {
            // THEY ARE RACED, NOT RANKED, and the handset in front of me is why.
            //
            // The order here used to be the answer: NNAPI if it builds, else XNNPACK,
            // else CPU, on the reasoning that dedicated silicon beats a CPU. It builds on
            // nearly every phone -- building is not the same as being quick. This device
            // took the NNAPI session happily and then spent 859 ms a frame in it, because
            // NNAPI had no accelerator that would take these ops and ran them on its own
            // fallback path. Nothing about the session says so; the only thing that knows
            // is the clock.
            //
            // So all three are built and timed on a frame-shaped blob, the quickest is
            // kept, the others are closed, and the winner is written down so the next
            // launch builds one session instead of three. A few seconds once, against
            // every frame for the life of the install.
            for ((name, make) in builders) {
                val candidate = try {
                    make()
                } catch (t: Throwable) {
                    Log.w(TAG, "provider $name unavailable", t)
                    continue
                }
                val ms = try {
                    timeOnePass(candidate)
                } catch (t: Throwable) {
                    Log.w(TAG, "provider $name built but would not run", t)
                    candidate.close()
                    continue
                }
                Log.i(TAG, "provider $name: ${"%.0f".format(ms)} ms a pass")
                if (chosen == null || ms < chosenMs) {
                    chosen?.close()
                    chosen = candidate
                    chosenName = name
                    chosenMs = ms
                } else {
                    candidate.close()
                }
            }
            if (chosenName.isNotEmpty()) {
                prefs?.edit()?.putString(key, chosenName)?.apply()
            }
        }

        session = chosen ?: throw IllegalStateException(
            "no execution provider would build a session for this model")
        provider = chosenName
        providerMs = chosenMs
        Log.i(TAG, "ONNX Runtime session on $provider, $threads threads")
    }

    /**
     * Milliseconds for one forward pass on zeros, which is all this comparison needs.
     *
     * Three passes and the FASTEST is taken, not the mean. The first pass through any
     * session includes whatever that provider does lazily -- NNAPI compiles the model for
     * the device, XNNPACK packs its weights -- and that cost is paid once at startup, not
     * on the frames an operator waits for. The mean would fold a one-off into the number
     * the choice is made on, which on the slowest-starting provider is exactly backwards.
     */
    private fun timeOnePass(candidate: OrtSession): Double {
        val name = candidate.inputNames.first()
        val shape = (candidate.inputInfo[name]!!.info as TensorInfo).shape
        var count = 1L
        for (d in shape) count *= if (d > 0) d else 1L
        val zeros = java.nio.FloatBuffer.allocate(count.toInt())
        var best = Double.MAX_VALUE
        OnnxTensor.createTensor(env, zeros, shape).use { input ->
            val feed = Collections.singletonMap(name, input)
            repeat(3) {
                val t0 = System.nanoTime()
                candidate.run(feed).use { }
                best = minOf(best, (System.nanoTime() - t0) / 1e6)
            }
        }
        return best
    }

    private val inputName: String = session.inputNames.first()

    /**
     * The input shape the export wants, TAKEN WHOLE from the model.
     *
     * It used to be read as one number -- shape[2] -- and squared, which is true of a
     * 640x640 export and of nothing else. model3's detector is exported at 480x640,
     * because that is the rectangle ultralytics feeds on the bench and a square one finds
     * pills that are not there (see model3/phone/engine.py). Told [1,3,480,480] while
     * Python sent 921,600 floats for [1,3,480,640], ONNX Runtime threw on every single
     * frame, the bridge caught it, and the handset showed "โมเดลผิดพลาด" with no count --
     * a failure whose cause was in Kotlin while everything it pointed at was in Python.
     *
     * Asking for the whole shape costs nothing and cannot be wrong for any export.
     */
    private val inputShape: LongArray =
        (session.inputInfo[inputName]!!.info as TensorInfo).shape

    /** Height and width of that shape, for callers that still think in one number. */
    private val size: Int = inputShape[2].toInt()

    /** Shapes of the tensors the last [forward] returned, as JSON: `[[1,37,4116],...]`. */
    @Volatile
    private var lastShapes: String = "[]"

    fun imgsz(): Int = size

    private companion object {
        const val TAG = "pillcount"

        /** Where the race's winner is remembered between launches. */
        const val KEY_PROVIDER = "provider"
    }

    fun shapes(): String = lastShapes

    /**
     * Run one frame. [chw] is a (1, 3, imgsz, imgsz) float32 blob in little-endian byte
     * order -- exactly `blobFromImage(...).tobytes()`.
     *
     * Returns every output tensor's raw float32 bytes, concatenated in the order the
     * graph declares them, with [shapes] saying where the boundaries are. One flat
     * `byte[]` rather than a `byte[][]`: Chaquopy converts a byte array to Python
     * `bytes` directly, where an array of them arrives as a jarray whose elements have
     * to be unwrapped one at a time. Python splits it with `np.frombuffer` and an
     * offset, which copies nothing.
     */
    fun forward(chw: ByteArray): ByteArray {
        val floats = ByteBuffer.wrap(chw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        OnnxTensor.createTensor(env, floats, inputShape).use { input ->
            session.run(Collections.singletonMap(inputName, input)).use { result ->
                // Take each FloatBuffer once and keep it. getFloatBuffer() hands out a
                // fresh view per call, so measuring with one and copying from another
                // is not obviously wrong -- but it is two reads of the same native
                // memory for no reason, and it only stays correct while that stays
                // true of the runtime.
                val buffers = ArrayList<java.nio.FloatBuffer>(result.size())
                val shapes = StringBuilder("[")
                var total = 0
                for (i in 0 until result.size()) {
                    val tensor = result[i] as OnnxTensor
                    if (i > 0) shapes.append(',')
                    (tensor.info as TensorInfo).shape.joinTo(shapes, ",", "[", "]")
                    val fb = tensor.floatBuffer
                    total += fb.remaining() * 4
                    buffers.add(fb)
                }
                shapes.append(']')

                val out = ByteArray(total)
                val sink = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                for (fb in buffers) sink.put(fb)

                lastShapes = shapes.toString()
                return out
            }
        }
    }

    fun close() {
        session.close()
    }
}
