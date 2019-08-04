import tensorflow as tf



tf.debugging.set_log_device_placement(True)

#==============학습데이터===========================
# 학습시간
x1_data = [54, 8, 30, 24, 46, 12, 20, 37, 40, 48]
# 토익점수
y_data = [800, 320, 600, 630, 700, 300, 920, 720, 700, 920]

#===============초기값 설정=========================
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#===============변수 설정===========================
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#=============== Linear Regression 에서 학습될 가설==============
hypothesis = W * X + b

#=================Linear Regression 에서 학습될 가상의 Cost Function ============
cost = tf.reduce_mean(tf.square(hypothesis - Y))


#================= Gradient Descent Algorithm 에서 Step ========
learning_rate = 0.0008

#================= TensorFlow에 내장된 Gradient Descent Optimizer=============
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


#================= 학습 시작======================
for step in range(5000):
    sess.run(optimizer, feed_dict={X:x1_data, Y:y_data})

    if step % 100 == 0:
        print (step, sess.run(cost, feed_dict={X:x1_data, Y:y_data}), sess.run(W), sess.run(b))

#=====================결과 예측===============================
print ("10시간: " + str(sess.run(hypothesis, feed_dict={X: 10})))
print ("15시간: " + str(sess.run(hypothesis, feed_dict={X: 15})))
print ("20시간: " + str(sess.run(hypothesis, feed_dict={X: 20})))
print ("25시간: " + str(sess.run(hypothesis, feed_dict={X: 25})))


