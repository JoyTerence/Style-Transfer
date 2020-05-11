import tensorflow as tf
from PIL import Image
import numpy as np
import vgg16
import matplotlib.pyplot as plt

image_height = 512
image_width = 512

content_image = Image.open(r"C:\Users\Joy.DESKTOP-M53NCFS\Documents\GitHub\Style-Transfer\images\content.png")
content_image = content_image.resize((image_width, image_height))
content = np.asarray(content_image, dtype='float32')
print (content.shape)

style_image = Image.open(r"C:\Users\Joy.DESKTOP-M53NCFS\Documents\GitHub\Style-Transfer\images\style.png")
style_image = style_image.resize((image_width, image_height))
style = np.asarray(style_image, dtype='float32')
print (style.shape)

content[:, :, 0] -= 103.939
content[:, :, 1] -= 116.779
content[:, :, 2] -= 123.68
content = content[:, :, ::-1]

style[:, :, 0] -= 103.939
style[:, :, 1] -= 116.779
style[:, :, 2] -= 123.68
style = style[:, :, ::-1]

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

vgg_16 = vgg16.VGG16()

content_layer = vgg_16.get_content_layer_tensors()

def calculate_content_loss():
    feed_dict_1 = vgg_16.create_feed_dict(image=content)
    content_layer_values = sess.run(content_layer, feed_dict=feed_dict_1)
    return tf.reduce_sum(tf.square(content_layer - content_layer_values))

def gram_matrix(tensor):
    #matrix =  tf.contrib.keras.backend.batch_flatten(tf.contrib.keras.backend.permute_dimensions(tensor, [2, 0, 1]))
    num_channels = tf.shape(tensor)[3]
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

def calculate_style_loss():
    feed_dict_2 = vgg_16.create_feed_dict(image=style)

    style_layers = vgg_16.get_style_layer_tensors()

    gram_layers = [gram_matrix(layer) for layer in style_layers]
    style_values = sess.run(gram_layers, feed_dict=feed_dict_2)

    layer_loss = []

    for gram_value, gram_layer in zip(style_values, gram_layers):

        value = tf.constant(gram_value)
        layer_loss.append(tf.reduce_sum(tf.square(value - gram_layer)))
    return (tf.reduce_mean(layer_loss))

def calculate_total_variation_loss(combination_image):
    a = tf.reduce_sum( tf.abs(combination_image[:, 1:, :, :] - combination_image[:, :-1, :, :]) )
    b = tf.reduce_sum( tf.abs(combination_image[:, :, 1:, :] - combination_image[:, :, :-1, :]) )
    return a + b


def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert pixels to bytes.
    image = image.astype(np.uint8)

    # Convert to a PIL-image and display it.
    display(Image.fromarray(image))

def plot_images(content_image, style_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

with vgg_16.graph.as_default():

    sess = tf.Session(graph=vgg_16.graph)
    loss = tf.Variable(0.0 , name='loss', dtype='float32')
    sess.run(tf.global_variables_initializer())

    loss += content_weight * calculate_content_loss()

    loss += (style_weight/ 5) * calculate_style_loss()

    loss += calculate_total_variation_loss(vgg_16.input)

    gradients = tf.gradients(loss, vgg_16.input)

    mixed_image = np.random.rand(*content.shape) + 128

    for i in range(120):

        feed_dict = vgg_16.create_feed_dict(image=mixed_image)

        grads = sess.run(gradients, feed_dict=feed_dict)

        grads = np.squeeze(grads)
        step_size = 10.0
        step_size_scaled = step_size / (np.std(grads) + 1e-8)
        # Update the image by following the gradient.
        mixed_image -= grads * step_size_scaled

        # Ensure the image has valid pixel-values between 0 and 255.
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Print a little progress-indicator.
        print(". ", end="")

        # Display status once every 10 iterations, and the last.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Plot the content-, style- and mixed-images.
            plot_images(content_image=content,
                        style_image=style,
                        mixed_image=mixed_image)

    print()
    print("Final image:")
    plot_image_big(mixed_image)
