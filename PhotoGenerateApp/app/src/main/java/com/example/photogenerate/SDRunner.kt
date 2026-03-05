package com.example.photogenerate

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer
import java.util.*
import org.pytorch.IValue
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.util.Log
import org.pytorch.DType
import kotlin.math.sqrt

class SDRunner(private val context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private lateinit var unetSession: OrtSession
    private lateinit var clipSession: OrtSession
    private lateinit var vaeSession: OrtSession

    init {
        //loadModels()
    }

    private fun getModelPath(modelName: String): String {
        val destinationFile = java.io.File(context.filesDir, modelName)
        val dataFileName = "$modelName.data"
        val destinationDataFile = java.io.File(context.filesDir, dataFileName)

        // Копируем .onnx, если его еще нет
        if (!destinationFile.exists()) {
            context.assets.open(modelName).use { input ->
                destinationFile.outputStream().use { output -> input.copyTo(output) }
            }
        }

        // Копируем .onnx.data, если он есть в assets (обязательно для больших моделей)
        val assetFiles = context.assets.list("") ?: emptyArray()
        if (assetFiles.contains(dataFileName) && !destinationDataFile.exists()) {
            context.assets.open(dataFileName).use { input ->
                destinationDataFile.outputStream().use { output -> input.copyTo(output) }
            }
        }

        return destinationFile.absolutePath
    }

    fun generate(prompt: String, updateProgress: (Int) -> Unit): Bitmap {
        // 1. CLIP (Загрузили -> Прогнали -> Удалили)
        val textEmbeds = runClipStep(prompt)
        System.gc() // Подсказываем системе очистить память
        val initialLatents = createRandomLatents()

        val latents = runUnetStep(initialLatents, textEmbeds, updateProgress)
        System.gc()
        System.runFinalization()
        Runtime.getRuntime().gc()
        textEmbeds.close()
        Thread.sleep(150)

        // 3. VAE (Загрузили -> Декодировали -> Удалили)
        //val testLatents = loadLatentsFromAssets(context)
        //val bitmap = runVaeStep(testLatents)
        val bitmap = runVaeStep(latents)
        System.gc()

        return bitmap
    }

    private fun runClipStep(prompt: String): OnnxTensor {
        Log.d("SD_DEBUG", "CLIP: Загрузка модели...")
        val options = OrtSession.SessionOptions() // Без NNAPI для CLIP (он маленький)
        val session = ortEnv.createSession(getModelPath("text_encoder_fp16.onnx"), options)

        val tokenIds = tokenizer.tokenize(prompt)
        val inputBuffer = java.nio.LongBuffer.wrap(tokenIds)
        val inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, longArrayOf(1, 77))

        val result = session.run(mapOf("input_ids" to inputTensor))
        val output = result[0] as OnnxTensor

        val fb = output.floatBuffer
        fb.rewind()
        val firstValues = FloatArray(10)
        fb.get(firstValues)

        Log.d("SD_DEBUG", "CLIP ПРОВЕРКА: ${firstValues.joinToString(", ")}")
        // ВАЖНО: Закрываем сессию, чтобы освободить 500МБ+ RAM
        session.close()
        Log.d("SD_DEBUG", "CLIP: Успешно выполнено")
        return output
    }
    private fun runUnetStep(latents: OnnxTensor, embeds: OnnxTensor, updateProgress: (Int) -> Unit): OnnxTensor {
        Log.d("SD_DEBUG", "UNET: Загрузка модели...")
        logTensorStats("START NOISE", latents)
        val options = OrtSession.SessionOptions().apply {
            // 1. Обязательно: Sequential Mode снижает пиковое потребление
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            // 2. Включаем агрессивное переиспользование буферов
            addConfigEntry("session.enable_memory_pattern", "1")
            setMemoryPatternOptimization(true)
            // 3. Отключаем лишние оптимизации графа, которые жрут память при инициализации
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            // 4. Важно для Android: используйте Арену
            addConfigEntry("session.use_device_allocator_for_initializers", "1")
        }

        val unetPath = getModelPath("unet_temp.onnx")
        val session = ortEnv.createSession(unetPath, options)

        var currentLatents = latents
        val steps = 4

        for (i in 0 until steps) {
            val tsBuffer = java.nio.ShortBuffer.allocate(1)
            val lcmSteps = floatArrayOf(999f, 759f, 499f, 259f)
            val currentTs = lcmSteps[i]
            tsBuffer.put(java.lang.Float.floatToFloat16(currentTs))
            tsBuffer.rewind()
            val timestep = OnnxTensor.createTensor(ortEnv, tsBuffer, longArrayOf(1), OnnxJavaType.FLOAT16)

            val inputs = mapOf(
                "sample" to currentLatents,
                "encoder_hidden_states" to embeds,
                "timestep" to timestep
            )

            val result = session.run(inputs)
            val noisePred = result[0] as OnnxTensor
            logTensorStats("UNET OUTPUT", noisePred)

            // Математика LCM
            val nextLatents = applyLcmStep(currentLatents, noisePred, i) // Создали новый тензор
            currentLatents.close()
            currentLatents = nextLatents
            result.close()
            logTensorStats("AFTER STEP $i", nextLatents)

            updateProgress(((i + 1) / steps.toFloat() * 100).toInt())
        }

        session.close() // ОСВОБОЖДАЕМ ПАМЯТЬ
        Log.d("SD_DEBUG", "UNET: Успешно выполнено")
        return currentLatents
    }

    /*private fun runVaeStep(latents: OnnxTensor): Bitmap {
        Log.d("SD_DEBUG", "VAE: Загрузка модели...")
        val scalingFactor = 0.18215f

        // 1. Получаем доступ к сырым данным (предполагаем FLOAT16)
        val floatBuffer = latents.floatBuffer // Если тензор FLOAT
        // Если тензор FLOAT16, используем shortBuffer:
        val shortBuffer = latents.shortBuffer
        val size = shortBuffer.capacity()
        shortBuffer.rewind()

        // 2. Создаем новый буфер для масштабированных данных
        val scaledBuffer = java.nio.ByteBuffer.allocateDirect(size * 2)
            .order(java.nio.ByteOrder.nativeOrder())
            .asShortBuffer()

        // 3. Цикл масштабирования
        for (i in 0 until size) {
            val h16 = shortBuffer.get()
            val rawValue = java.lang.Float.float16ToFloat(h16)

            // КЛЮЧЕВОЙ МОМЕНТ: Делим на 0.18215 ПЕРЕД VAE
            val scaledValue = rawValue / scalingFactor

            scaledBuffer.put(java.lang.Float.floatToFloat16(scaledValue))
        }
        scaledBuffer.rewind()

        // 4. Создаем новый тензор для VAE
        val shape = latents.info.shape // Обычно [1, 4, 64, 64]
        val vaeInputTensor = OnnxTensor.createTensor(ortEnv, scaledBuffer, shape, OnnxJavaType.FLOAT16)

        logTensorStats("BEFORE VAE",vaeInputTensor)
        val module = LiteModuleLoader.load(getModelPath("vae_decoder.ptl"))

        // 3. Выполнениемодель
        val shortBuffer = ByteBuffer.allocateDirect(scaledBuffer.size * 2) // 2 байта на short
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
        shortBuffer.put(scaledBuffer)
        shortBuffer.rewind()

// 3. Создаем тензор, ЯВНО указывая тип FLOAT16
        val inputShape = longArrayOf(1, 4, 64, 64)
        val inputTensor = Tensor.fromBlob(shortBuffer, inputShape, DType.FLOAT16)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

// 4. Извлечение данных [1, 3, 512, 512]
        val scores = outputTensor.dataAsFloatArray

        /* 1. Создаем локальную сессию для VAE
        val options = OrtSession.SessionOptions().apply {
            // Отключаем кэширование памяти в самом ONNX
            addConfigEntry("session.enable_cpu_mem_arena", "0")
            // Устанавливаем минимальный уровень оптимизации, чтобы не плодить буферы
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        }
        val vaePath = getModelPath("vae_decoder_fp16.onnx")
        val session = ortEnv.createSession(vaePath, options)

        Log.d("SD_DEBUG", "VAE: Декодирование латентов...")

        // 2. Запуск декодера
        // Важно: имя входа "latent_sample" должно совпадать с вашей моделью
        val result = session.run(mapOf("latent_sample" to vaeInputTensor))

        // Получаем ShortBuffer, так как модель FP16 (Float16)
        val outputTensor = result[0] as OnnxTensor
        val outputRaw = outputTensor.shortBuffer
        outputRaw.rewind()
        Log.d("SD_DEBUG", "Реальный тип выхода VAE: ${outputTensor.info.type}")
        logTensorStats("AFTER VAE",outputTensor)*/

        // 3. Подготовка Bitmap
        val width = 512
        val height = 512
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // В тензоре данные обычно лежат в формате [1, 3, 512, 512] (NCHW)
        // Это значит: сначала весь канал Red, потом Green, потом Blue
        val channelSize = width * height

        Log.d("SD_DEBUG", "VAE: Сборка финального изображения...")

        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x

                // Извлекаем значения и конвертируем из Float16 в Float32
                // Внутри цикла y/x используй абсолютный индекс:
                // Перед тем как переводить в 0..255:
                val scaleFactor = 0.5f //0.18215f
                val rRaw = java.lang.Float.float16ToFloat(outputRaw.get(index)) * scaleFactor
                val gRaw = java.lang.Float.float16ToFloat(outputRaw.get(index + channelSize)) * scaleFactor
                val bRaw = java.lang.Float.float16ToFloat(outputRaw.get(index + 2 * channelSize)) * scaleFactor
// И убедись, что index рассчитывается как (y * width + x)

                // Нормализация: из [-1, 1] в [0, 255]
                // Формула: (x + 1) * 127.5
                val r = ((rRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)
                val g = ((gRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)
                val b = ((bRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)

                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }

        // 4. ОЧИСТКА: Закрываем всё, чтобы вернуть память системе
        result.close()
        session.close()

        Log.d("SD_DEBUG", "VAE: Завершено, сессия закрыта.")
        return bitmap
    }*/

    private fun runVaeStep(latents: OnnxTensor): Bitmap {
        Log.d("SD_DEBUG", "VAE: Начало работы (PyTorch Mobile FP16)")
        val scalingFactor = 0.18215f

        // 1. Подготовка входных данных из ONNX-тензора в PyTorch-тензор
        val shortBuffer = latents.shortBuffer
        val size = shortBuffer.capacity()
        shortBuffer.rewind()
        val latentsFloatArray = FloatArray(16384)

        for (i in 0 until size) {
            val h16 = shortBuffer.get()
            val rawValue = java.lang.Float.float16ToFloat(h16)

            // Делим на коэффициент скейлинга
            latentsFloatArray[i] = rawValue / scalingFactor
        }

        // 2. Загрузка модели PyTorch Lite
        val module = LiteModuleLoader.load(getModelPath( "vae_decoder.ptl"))

        val inputShape = longArrayOf(1, 4, 64, 64)
        val inputTensor = Tensor.fromBlob(latentsFloatArray, inputShape)

        // 3. Прогон модели
        Log.d("SD_DEBUG", "VAE: Выполнение forward...")
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        // 5. Сборка Bitmap
        val width = 512
        val height = 512
        val channelSize = width * height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        Log.d("SD_DEBUG", "VAE: Сборка финального изображения...")

        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x

                // Извлекаем каналы (NCHW) и конвертируем FP16 -> Float32
                val rRaw = scores[index]
                val gRaw = scores[index + channelSize]
                val bRaw = scores[index + 2 * channelSize]

                // Нормализация: из [-1, 1] в [0, 255]
                // Добавляем coerceIn, чтобы "выбросы" не ломали картинку
                val r = ((rRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)
                val g = ((gRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)
                val b = ((bRaw + 1f) * 127.5f).toInt().coerceIn(0, 255)

                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }

        Log.d("SD_DEBUG", "VAE: Завершено успешно!")
        return bitmap
    }
    private val tokenizer = Tokenizer(context)

    private fun logTensorStats(name: String, tensor: OnnxTensor) {
        val buffer = tensor.shortBuffer
        buffer.rewind()
        val capacity = buffer.capacity()

        var sum = 0f
        var sumSq = 0f
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        val first5 = FloatArray(5)

        for (i in 0 until capacity) {
            val v = java.lang.Float.float16ToFloat(buffer.get())
            sum += v
            sumSq += v * v
            if (v < min) min = v
            if (v > max) max = v
            if (i < 5) first5[i] = v
        }

        val mean = sum / capacity
        val std = kotlin.math.sqrt((sumSq / capacity) - (mean * mean))

        Log.d("SD_STATS", "--- $name ---")
        Log.d("SD_STATS", "Mean: %.4f, Std: %.4f".format(mean, std))
        Log.d("SD_STATS", "Min: %.4f, Max: %.4f".format(min, max))
        Log.d("SD_STATS", "First 5: ${first5.joinToString(", ")}")
    }

    fun loadLatentsFromAssets(context: Context): OnnxTensor {
        val fileName = "latents_test.bin"
        val bytes = context.assets.open(fileName).readBytes()

        // Создаем Direct ByteBuffer для Float32 (4 байта на число)
        // При загрузке из Assets:
        val shortBuffer = java.nio.ByteBuffer.allocateDirect(bytes.size / 2) // в 2 раза меньше места чем для Float32
            .order(java.nio.ByteOrder.nativeOrder())
            .asShortBuffer()
        // Читаем float из файла и жмем в float16
        val byteBuffer = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        while (byteBuffer.hasRemaining()) {
            val f32 = byteBuffer.float
            shortBuffer.put(java.lang.Float.floatToFloat16(f32))
        }
        shortBuffer.rewind()
        val shape = longArrayOf(1, 4, 64, 64)
        val testLatents = OnnxTensor.createTensor(ortEnv, shortBuffer, shape, OnnxJavaType.FLOAT16)
        return testLatents
    }
    private fun applyLcmStep(latents: OnnxTensor, noisePred: OnnxTensor, step: Int): OnnxTensor {
        val lBuffer = latents.shortBuffer
        val nBuffer = noisePred.shortBuffer
        val size = lBuffer.capacity()

        // Используй allocateDirect для стабильности в ONNX Runtime
        val bb = java.nio.ByteBuffer.allocateDirect(size * 2).order(java.nio.ByteOrder.nativeOrder())
        val outputBuffer = bb.asShortBuffer()

        // Коэффициенты текущего шага (t)
        val alphas = doubleArrayOf(0.00466009508818388, 0.05221289023756981, 0.27766942977905273, 0.6589752435684204)
        val sigmas = doubleArrayOf(0.9953399049118161, 0.9477871097624302, 0.7223305702209473, 0.3410247564315796)

        // Коэффициенты ПРЕДЫДУЩЕГО шага (t_prev) для подмешивания шума
        // На последнем шаге (step=3) эти значения не будут использоваться
        val alphasPrev = doubleArrayOf( 0.05221289023756981, 0.27766942977905273, 0.6589752435684204, 0.9996)
        val sigmasPrev = doubleArrayOf(0.9477871097624302, 0.7223305702209473, 0.3410247564315796, 0.0292)

        val a_t = sqrt(alphas[step])
        val s_t = sqrt(sigmas[step])
        val a_prev = sqrt(alphasPrev[step])
        val s_prev = sqrt(sigmasPrev[step])

        // Расчет c_skip и c_out на основе "граничных условий" LCM
        // sigma_data в LCM обычно 0.5

        val random = java.util.Random()
        lBuffer.rewind()
        nBuffer.rewind()

        for (i in 0 until size) {
            val x_t = java.lang.Float.float16ToFloat(lBuffer.get())
            val epsilon = java.lang.Float.float16ToFloat(nBuffer.get())

            // 1. Предсказываем чистый образец x0 (пункт 4 в исходнике)
            var pred_x0 = (x_t - s_t * epsilon) / a_t

            // 2. Клампинг (пункт 5 в исходнике) - УБИРАЕТ ПЯТНА
            //pred_x0 = pred_x0.coerceIn(-1.0f, 1.0f)

            // 3. Denoised результат (пункт 6 в исходнике)
            val denoised =  pred_x0

            // 4. Подмешивание шума для многошаговой генерации (пункт 7 в исходнике)
            val resultValue: Double
            if (step < 3) {
                val z = random.nextGaussian().toFloat()
                resultValue = (a_prev * denoised) + (s_prev * z)
            } else {
                resultValue = denoised // На финальном шаге шум не нужен
            }

            outputBuffer.put(java.lang.Float.floatToFloat16(resultValue.toFloat()))
        }

        outputBuffer.rewind()
        // Убедись, что форма тензора совпадает с моделью (NCHW)
        return OnnxTensor.createTensor(ortEnv, outputBuffer, longArrayOf(1, 4, 64, 64), OnnxJavaType.FLOAT16)
    }


    // Вспомогательная функция для возведения в степень
    private fun Float.pow(n: Int): Float = Math.pow(this.toDouble(), n.toDouble()).toFloat()

    private fun convertToFloat16Tensor(tensor: OnnxTensor): OnnxTensor {
        val shape = tensor.info.shape
        val floatBuffer = tensor.floatBuffer
        val size = floatBuffer.capacity()
        val shortBuffer = java.nio.ShortBuffer.allocate(size)

        floatBuffer.rewind()
        for (i in 0 until size) {
            shortBuffer.put(java.lang.Float.floatToFloat16(floatBuffer.get()))
        }
        shortBuffer.rewind()
        return OnnxTensor.createTensor(ortEnv, shortBuffer, shape, OnnxJavaType.FLOAT16)
    }

    private fun createRandomLatents(): OnnxTensor {
        val size = 1 * 4 * 64 * 64
        val shape = longArrayOf(1, 4, 64, 64)
        val buffer = java.nio.ByteBuffer.allocateDirect(size * 2).order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()
        val random = java.util.Random()

        for (i in 0 until size) {
            // Стандартное нормальное распределение (Mean 0, Std 1)
            val gaussian = random.nextGaussian().toFloat()

            // ВАЖНО: Убедись, что ты НЕ умножаешь его ни на что лишнее здесь.
            // Но проверь, не слишком ли "тихий" шум.
            // Попробуй gaussian * 1.0f (стандарт)

            buffer.put(java.lang.Float.floatToFloat16(gaussian))
        }
        buffer.rewind()
        return OnnxTensor.createTensor(ortEnv, buffer, shape, OnnxJavaType.FLOAT16)
    }

}
