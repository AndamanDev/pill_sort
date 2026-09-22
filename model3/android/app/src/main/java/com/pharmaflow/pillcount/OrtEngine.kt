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

    /** Where the crash breadcrumb lives, and whether it has been earned yet.
     *
     * DECLARED ABOVE THE init BLOCK, and that is not tidiness. Kotlin runs property
     * initialisers and init blocks in SOURCE ORDER, so `= null` written below the block
     * would fire after it and quietly undo the assignment the block made -- healthy()
     * would then rub out nothing, and the next launch would blame a provider that had in
     * fact worked perfectly.
     */
    private var prefs: SharedPreferences? = null
    private var pendingKey: String = ""
    @Volatile
    private var proven = false

    init {
        // Four threads at most. A phone's big cores are four; asking for every core lets
        // ORT schedule onto the little ones too, where the per-thread sync costs more
        // than the work they contribute.
        val threads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)

        // WHAT KILLED US LAST TIME, if anything did.
        //
        // The race below catches a provider that is SLOW. It cannot catch one that is
        // FATAL: a vendor driver that segfaults takes the process with it, and nothing in
        // Kotlin runs afterwards -- no catch, no log line, no chance to try the next one.
        // Shipping a quantised graph to this bench did exactly that. The app closed the
        // instant the loading screen finished, which is when the first REAL frame goes
        // through, and the only evidence was that it had gone.
        //
        // So the name of whatever is about to be trusted is written down BEFORE it is
        // used, and rubbed out by [healthy] once a real frame has been through it and
        // come back. Finding it still there on the next launch is the process saying it
        // did not survive that choice, in the only way a process that died can.
        val pendingKey = "$KEY_PENDING-$stamp"
        val fatal = prefs?.getString(pendingKey, null)
        if (fatal != null) {
            Log.w(TAG, "provider $fatal did not survive the last run; skipping it")
            prefs.edit().remove(pendingKey).remove("$KEY_PROVIDER-$stamp").apply()
        }

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
        if (fatal != null) builders.remove(fatal)
        if (builders.isEmpty()) {
            // Everything this build knows how to run has now taken the process down. The
            // graph itself is the suspect at that point, not the provider, and the honest
            // thing is to say so rather than to loop.
            throw IllegalStateException(
                "every execution provider has crashed this device on this model")
        }

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
        // DOWN BEFORE THE FIRST REAL FRAME, not after. The point of the breadcrumb is to
        // survive a crash, so it has to be on disk before the thing that might crash.
        this.prefs = prefs
        this.pendingKey = pendingKey
        prefs?.edit()?.putString(pendingKey, chosenName)?.apply()
        Log.i(TAG, "ONNX Runtime session on $provider, $threads threads"
                + if (fatal != null) " (after $fatal crashed the last run)" else "")
    }

    /**
     * Milliseconds for one forward pass on zeros, which is all this comparison needs.
     *
     * TWO passes and the FASTEST is taken, not the mean. The first pass through any
     * session includes whatever that provider does lazily -- NNAPI compiles the model for
     * the device, XNNPACK packs its weights -- and that cost is paid once at startup, not
     * on the frames an operator waits for. The mean would fold a one-off into the number
     * the choice is made on, which on the slowest-starting provider is exactly backwards.
     *
     * It was three, which is one warm-up and two measurements of the same thing. On a
     * desktop that is free; on this bench a pass is 556 ms, so the third was a second and
     * a half of loading screen per race for a number that never changed the answer.
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
            repeat(2) {
                val t0 = System.nanoTime()
                candidate.run(feed).use { }
                best = minOf(best, (System.nanoTime() - t0) / 1e6)
            }
        }
        return best
    }

    /**
     * A real frame has been through this provider and come back. Rub the breadcrumb out.
     *
     * Called by the activity from the analyser, not from [forward] itself: forward returns
     * the moment the graph does, and what has to be proved is that the whole frame
     * survived -- the copy out of native memory included, which is where a driver that
     * mis-reports a tensor's size takes the process down.
     */
    fun healthy() {
        if (proven) return
        proven = true
        prefs?.edit()?.remove(pendingKey)?.apply()
        Log.i(TAG, "provider $provider survived a real frame")
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

    /**
     * The export's input HEIGHT and WIDTH, for the letterbox on the Python side.
     *
     * [imgsz] returns one number and is kept for callers that still think in squares. This
     * detector is not square -- it is exported at the rectangle the bench feeds, and the
     * two numbers are not interchangeable. Python used to assume 480x640 from a constant
     * in engine.py, which was true of the file that happened to be shipped and of nothing
     * else: swapping in a smaller export to buy frame rate would have letterboxed every
     * frame to the wrong rectangle and fed the graph a tensor it would refuse, on every
     * frame, with the fault surfacing in Python as a shape error about a file Kotlin
     * chose. Asking the graph costs one field read at startup and cannot go stale.
     */
    fun inputHeight(): Int = inputShape[2].toInt()

    fun inputWidth(): Int = inputShape[3].toInt()

    private companion object {
        const val TAG = "pillcount"

        /** Where the race's winner is remembered between launches. */
        const val KEY_PROVIDER = "provider"

        /** And where the one currently being trusted is, until it has proved itself. */
        const val KEY_PENDING = "pending"
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
