import { ref, onMounted, onUnmounted } from 'vue'
import gsap from 'gsap'

export function useLotteryBall() {
  const ballRefs = ref<HTMLElement[]>([])

  // 掉落动画
  const animateDrop = (elements: HTMLElement[], options?: {
    delay?: number
    stagger?: number
    onComplete?: () => void
  }) => {
    const { delay = 0, stagger = 0.15, onComplete } = options || {}

    gsap.fromTo(elements,
      {
        y: -300,
        opacity: 0,
        scale: 0.5
      },
      {
        y: 0,
        opacity: 1,
        scale: 1,
        duration: 0.8,
        ease: 'bounce.out',
        stagger,
        delay,
        onComplete: () => {
          // 掉落完成后的发光效果
          gsap.to(elements, {
            boxShadow: '0 0 30px currentColor',
            duration: 0.3,
            yoyo: true,
            repeat: 1
          })
          onComplete?.()
        }
      }
    )
  }

  // 悬停效果
  const onHover = (element: HTMLElement) => {
    gsap.to(element, {
      scale: 1.1,
      boxShadow: '0 0 40px currentColor',
      duration: 0.3
    })
  }

  // 离开效果
  const onLeave = (element: HTMLElement) => {
    gsap.to(element, {
      scale: 1,
      boxShadow: '0 0 20px currentColor',
      duration: 0.3
    })
  }

  // 点击旋转
  const onClick = (element: HTMLElement) => {
    gsap.to(element, {
      rotation: '+=360',
      duration: 0.5,
      ease: 'power2.out'
    })
  }

  return {
    ballRefs,
    animateDrop,
    onHover,
    onLeave,
    onClick
  }
}
