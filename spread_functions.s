	.file	"spread_functions.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,"axG",@progbits,_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.type	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, @function
_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv:
.LFB11341:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L6
	movq	32(%rbx), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	$0, (%rbx)
	movl	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movl	$0, 24(%rbx)
	movq	$0, 32(%rbx)
.L6:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11341:
	.size	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, .-_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"vector::_M_realloc_append"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB13:
	.text
.LHOTB13:
	.p2align 4
	.globl	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff:
.LFB10556:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA10556
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	movq	%rdx, %r15
	subq	$640, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	8(%rsi), %rax
	movq	(%rsi), %r12
	movq	%r10, -520(%rbp)
	movq	%rdi, -648(%rbp)
	movq	%rsi, -400(%rbp)
	vmovss	%xmm0, -532(%rbp)
	vmovss	%xmm1, -544(%rbp)
	vmovss	%xmm2, -404(%rbp)
	vmovss	%xmm3, -548(%rbp)
	movq	%rax, -424(%rbp)
	movq	8(%rdx), %rax
	movq	%rax, -376(%rbp)
	movq	(%rdx), %rcx
	movq	$0, -592(%rbp)
	movq	$0, -528(%rbp)
	movq	$0, -432(%rbp)
	subq	%rcx, %rax
	movq	%rcx, -384(%rbp)
	movq	%rax, %r14
	sarq	$4, %r14
	movq	%r14, -440(%rbp)
	testq	%rax, %rax
	je	.L10
	xorl	%r13d, %r13d
	jmp	.L20
	.p2align 6
	.p2align 4
	.p2align 3
.L183:
	vmovdqu	(%rcx), %xmm0
	addq	$16, %rax
	incq	%r13
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -528(%rbp)
	cmpq	%r14, %r13
	jnb	.L10
.L184:
	movq	(%r15), %rcx
.L20:
	movq	%r13, %rax
	movq	-592(%rbp), %rbx
	salq	$4, %rax
	addq	%rax, %rcx
	movq	-528(%rbp), %rax
	cmpq	%rbx, %rax
	jne	.L183
	movq	-432(%rbp), %rax
	movq	%rbx, %rsi
	movabsq	$576460752303423487, %rbx
	subq	%rax, %rsi
	movq	%rsi, %rax
	sarq	$4, %rax
	cmpq	%rbx, %rax
	je	.L173
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L15
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rbx
.L16:
	movq	%rbx, %rdi
	movq	%rsi, -368(%rbp)
	movq	%rcx, -360(%rbp)
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	movq	-360(%rbp), %rcx
	movq	-368(%rbp), %rsi
	movq	-432(%rbp), %rdx
	vmovdqu	(%rcx), %xmm0
	vmovdqu	%xmm0, (%rax,%rsi)
	movq	-528(%rbp), %rsi
	cmpq	%rdx, %rsi
	je	.L110
	movq	%rax, %rcx
	.p2align 5
	.p2align 4
	.p2align 3
.L18:
	vmovdqu	(%rdx), %xmm0
	addq	$16, %rdx
	addq	$16, %rcx
	vmovdqu	%xmm0, -16(%rcx)
	cmpq	%rdx, %rsi
	jne	.L18
	movq	-528(%rbp), %rdx
	movq	-432(%rbp), %rdi
	subq	%rdi, %rdx
	addq	%rax, %rdx
.L17:
	addq	$16, %rdx
	movq	%rdx, -528(%rbp)
	movq	-432(%rbp), %rdx
	testq	%rdx, %rdx
	je	.L19
	movq	-592(%rbp), %rsi
	movq	%rdx, %rdi
	movq	%rax, -360(%rbp)
	subq	%rdx, %rsi
	call	_ZdlPvm@PLT
	movq	-360(%rbp), %rax
.L19:
	addq	%rax, %rbx
	incq	%r13
	movq	%rax, -432(%rbp)
	movq	%rbx, -592(%rbp)
	cmpq	%r14, %r13
	jb	.L184
.L10:
	movl	$8, %edi
.LEHB1:
	call	_Znwm@PLT
.LEHE1:
	movq	%r14, (%rax)
	movq	%rax, -640(%rbp)
	addq	$8, %rax
	vmovq	%r12, %xmm3
	movq	%rax, -624(%rbp)
	movl	$0, -152(%rbp)
	movq	-424(%rbp), %rax
	vpinsrq	$1, %rax, %xmm3, %xmm3
	imulq	%r12, %rax
	vmovdqa	%xmm3, -688(%rbp)
	vmovdqa	%xmm3, -176(%rbp)
	movq	%rax, %rbx
	testq	%rax, %rax
	jne	.L185
	movl	$0, -552(%rbp)
	movq	$0, -656(%rbp)
	movq	$0, -392(%rbp)
	movq	$0, -664(%rbp)
.L21:
	movq	-384(%rbp), %rbx
	cmpq	%rbx, -376(%rbp)
	je	.L31
	movq	(%r15), %rcx
	xorl	%esi, %esi
	movl	$1, %r8d
	.p2align 4
	.p2align 3
.L32:
	movq	8(%rcx), %rax
	movq	-392(%rbp), %rbx
	imulq	%r12, %rax
	addq	(%rcx), %rax
	leaq	63(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$6, %rdx
	leaq	(%rbx,%rdx,8), %rdi
	cqto
	shrq	$58, %rdx
	addq	%rdx, %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L186
	incq	%rsi
	shlx	%rax, %r8, %rax
	addq	$16, %rcx
	orq	%rax, (%rdi)
	cmpq	%r14, %rsi
	jb	.L32
.L31:
	cmpq	$-1, %r14
	je	.L187
.L25:
	vmovss	.LC2(%rip), %xmm3
	movq	-624(%rbp), %rax
	movq	$0, -360(%rbp)
	vdivss	-532(%rbp), %xmm3, %xmm3
	movq	%rax, -632(%rbp)
	movq	-440(%rbp), %rax
	vmovss	%xmm3, -540(%rbp)
	vmovss	.LC2(%rip), %xmm3
	vdivss	-404(%rbp), %xmm3, %xmm3
	vmovss	%xmm3, -536(%rbp)
	.p2align 4
	.p2align 3
.L33:
	leaq	-336(%rbp), %rbx
	movq	%rax, -560(%rbp)
	movq	%rbx, -512(%rbp)
	cmpq	%rax, -360(%rbp)
	jnb	.L188
	.p2align 4
	.p2align 3
.L60:
	movq	-360(%rbp), %rax
	movq	-432(%rbp), %rbx
	movq	-400(%rbp), %rsi
	movq	-512(%rbp), %rdi
	salq	$4, %rax
	addq	%rbx, %rax
	movq	(%rax), %r13
	movq	8(%rax), %rbx
	movq	%r13, %rdx
	movq	%rbx, %rcx
	vzeroupper
.LEHB2:
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	leaq	-112(%rbp), %rax
	movl	%r13d, %r9d
	leaq	_ZL5MOVES(%rip), %r13
	movl	%ebx, %r8d
	movq	%rax, -448(%rbp)
	movq	%rax, %rdx
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	leaq	4(%r13), %r11
	movl	$1, %r14d
	jmp	.L39
	.p2align 4
	.p2align 3
.L190:
	movl	%ecx, %esi
	orl	%edi, %esi
	js	.L34
	movl	-424(%rbp), %esi
	cmpl	%esi, %ecx
	jge	.L34
	movslq	%ecx, %rcx
	movq	%r12, %rsi
	movq	-392(%rbp), %r15
	movl	%edi, (%rdx)
	imulq	%rcx, %rsi
	movslq	%edi, %rdi
	leaq	63(%rdi,%rsi), %r10
	addq	%rdi, %rsi
	cmovns	%rsi, %r10
	sarq	$6, %r10
	leaq	(%r15,%r10,8), %r15
	movq	%rsi, %r10
	sarq	$63, %r10
	shrq	$58, %r10
	addq	%r10, %rsi
	andl	$63, %esi
	subq	%r10, %rsi
	jns	.L35
	addq	$64, %rsi
	subq	$8, %r15
.L35:
	shlx	%rsi, %r14, %rsi
	andq	(%r15), %rsi
	jne	.L36
	movq	-400(%rbp), %r15
	imulq	96(%r15), %rcx
	leaq	63(%rdi,%rcx), %rsi
	addq	%rdi, %rcx
	movq	112(%r15), %rdi
	cmovns	%rcx, %rsi
	sarq	$6, %rsi
	leaq	(%rdi,%rsi,8), %rdi
	movq	%rcx, %rsi
	sarq	$63, %rsi
	shrq	$58, %rsi
	addq	%rsi, %rcx
	andl	$63, %ecx
	subq	%rsi, %rcx
	jns	.L37
	addq	$64, %rcx
	subq	$8, %rdi
.L37:
	movl	$1, %esi
	shlx	%rax, %rsi, %r10
	shlx	%rcx, %rsi, %rsi
	andq	(%rdi), %rsi
	je	.L36
	incq	%rax
	orq	%r10, %rbx
	addq	$4, %rdx
	cmpq	$8, %rax
	je	.L189
.L39:
	movl	0(%r13,%rax,8), %edi
	movl	(%r11,%rax,8), %ecx
	addl	%r9d, %edi
	addl	%r8d, %ecx
	movl	%ecx, 32(%rdx)
	cmpl	%r12d, %edi
	jl	.L190
.L34:
	movl	%r9d, (%rdx)
	movl	%r8d, 32(%rdx)
.L36:
	shlx	%rax, %r14, %rcx
	incq	%rax
	addq	$4, %rdx
	andn	%rbx, %rcx, %rbx
	cmpq	$8, %rax
	jne	.L39
.L189:
	testq	%rbx, %rbx
	jne	.L191
.L59:
	incq	-360(%rbp)
	movq	-440(%rbp), %rbx
	movq	-360(%rbp), %rax
	cmpq	%rbx, %rax
	jne	.L60
	movq	-624(%rbp), %rax
	movq	-632(%rbp), %rbx
	cmpq	%rbx, %rax
	je	.L62
	movq	-560(%rbp), %rbx
	addq	$8, %rax
	movq	%rbx, -8(%rax)
	movq	%rax, -624(%rbp)
.L64:
	movq	-440(%rbp), %rax
	movq	%rax, -360(%rbp)
	cmpq	%rbx, %rax
	je	.L63
	movq	%rbx, %rax
	movq	%rbx, -440(%rbp)
	jmp	.L33
	.p2align 4
	.p2align 3
.L191:
	movq	-520(%rbp), %rdx
	movq	-400(%rbp), %r14
	movslq	-80(%rbp), %r8
	movslq	-112(%rbp), %r9
	vmovsd	8(%rdx), %xmm4
	vmovsd	16(%rdx), %xmm3
	vmovsd	56(%rdx), %xmm7
	movq	%r8, %rax
	movq	32(%r14), %rdi
	movq	152(%r14), %rcx
	movq	168(%r14), %rsi
	movq	232(%r14), %r15
	movq	208(%r14), %r13
	leaq	(%rsi,%r9), %r10
	vmovsd	%xmm4, -488(%rbp)
	vmovsd	%xmm3, -496(%rbp)
	vmovsd	24(%rdx), %xmm4
	vmovsd	64(%rdx), %xmm3
	vmovsd	%xmm7, -480(%rbp)
	movq	16(%r14), %rdx
	imulq	%rdx, %rax
	vmovsd	%xmm4, -504(%rbp)
	vmovsd	%xmm3, -472(%rbp)
	addq	%r9, %rax
	movzwl	(%rdi,%rax,2), %eax
	movw	%ax, -404(%rbp)
	movq	%r8, %rax
	imulq	%rcx, %rax
	movzbl	(%r10,%rax), %eax
	movq	%r8, %r10
	imulq	%r15, %r8
	addq	%r9, %r8
	movb	%al, -344(%rbp)
	movq	192(%r14), %rax
	movq	%r8, -456(%rbp)
	movslq	-76(%rbp), %r8
	imulq	%rax, %r10
	leaq	(%r10,%r9), %r11
	movslq	-108(%rbp), %r9
	movq	%rdx, %r10
	imulq	%r8, %r10
	movq	%r11, -416(%rbp)
	movq	248(%r14), %r11
	leaq	(%r10,%r9), %r14
	movq	%rcx, %r10
	imulq	%r8, %r10
	movq	%r11, -368(%rbp)
	leaq	(%rsi,%r9), %r11
	movq	%r14, -464(%rbp)
	movq	%r13, %r14
	movzbl	(%r11,%r10), %r10d
	movq	%r14, -376(%rbp)
	movb	%r10b, -343(%rbp)
	movq	%rax, %r10
	imulq	%r8, %r10
	imulq	%r15, %r8
	addq	%r9, %r10
	vmovss	0(%r13,%r10,4), %xmm2
	movq	-368(%rbp), %r13
	addq	%r9, %r8
	movq	%rdx, %r10
	movslq	-104(%rbp), %r9
	vmovss	0(%r13,%r8,4), %xmm1
	movslq	-72(%rbp), %r8
	leaq	(%rsi,%r9), %r11
	imulq	%r8, %r10
	addq	%r9, %r10
	movzwl	(%rdi,%r10,2), %r10d
	movw	%r10w, -408(%rbp)
	movq	%rcx, %r10
	imulq	%r8, %r10
	movzbl	(%r11,%r10), %r10d
	movb	%r10b, -342(%rbp)
	movq	%rax, %r10
	imulq	%r8, %r10
	imulq	%r15, %r8
	addq	%r9, %r10
	addq	%r8, %r9
	movslq	-68(%rbp), %r8
	movq	%r10, -568(%rbp)
	movq	%rdx, %r10
	movq	%r9, -576(%rbp)
	movslq	-100(%rbp), %r9
	imulq	%r8, %r10
	leaq	(%r10,%r9), %r11
	movq	%rcx, %r10
	imulq	%r8, %r10
	movq	%r11, -584(%rbp)
	leaq	(%rsi,%r9), %r11
	movzbl	(%r11,%r10), %r10d
	movq	%rcx, %r11
	movb	%r10b, -341(%rbp)
	movq	%rax, %r10
	imulq	%r8, %r10
	imulq	%r15, %r8
	addq	%r9, %r10
	addq	%r9, %r8
	movslq	-64(%rbp), %r9
	vmovss	(%r14,%r10,4), %xmm6
	movslq	-96(%rbp), %r10
	vmovss	0(%r13,%r8,4), %xmm4
	movq	%rdx, %r8
	imulq	%r9, %r11
	leaq	(%rsi,%r10), %r13
	imulq	%r9, %r8
	movzbl	0(%r13,%r11), %r11d
	addq	%r10, %r8
	movzwl	(%rdi,%r8,2), %r8d
	movb	%r11b, -340(%rbp)
	movq	%rax, %r11
	imulq	%r9, %r11
	imulq	%r15, %r9
	vmovd	%r8d, %xmm3
	addq	%r10, %r11
	addq	%r10, %r9
	movq	%r11, -600(%rbp)
	movslq	-92(%rbp), %r10
	movq	%rdx, %r11
	movq	%r9, -608(%rbp)
	movslq	-60(%rbp), %r9
	imulq	%r9, %r11
	leaq	(%r11,%r10), %r14
	movq	%rcx, %r11
	imulq	%r9, %r11
	movq	%r14, -616(%rbp)
	leaq	(%rsi,%r10), %r14
	movzbl	(%r14,%r11), %r11d
	movb	%r11b, -339(%rbp)
	movq	%rax, %r11
	movq	-368(%rbp), %r13
	movq	-376(%rbp), %r14
	imulq	%r9, %r11
	movq	%r15, -384(%rbp)
	movq	-616(%rbp), %r8
	vmovd	-404(%rbp), %xmm0
	imulq	%r15, %r9
	vmovd	-408(%rbp), %xmm9
	addq	%r10, %r11
	addq	%r10, %r9
	vmovss	(%r14,%r11,4), %xmm7
	movslq	-88(%rbp), %r11
	movq	%rcx, %r14
	vmovss	0(%r13,%r9,4), %xmm5
	movslq	-56(%rbp), %r9
	movq	%rdx, %r10
	vpinsrw	$1, (%rdi,%r8,2), %xmm3, %xmm3
	leaq	(%rsi,%r11), %r13
	imulq	%r9, %r14
	imulq	%r9, %r10
	movzbl	0(%r13,%r14), %r14d
	addq	%r11, %r10
	movzwl	(%rdi,%r10,2), %r10d
	movb	%r14b, -338(%rbp)
	movq	%rax, %r14
	imulq	%r9, %r14
	imulq	%r15, %r9
	movq	-464(%rbp), %r15
	vmovd	%r10d, %xmm8
	movq	-568(%rbp), %r10
	addq	%r11, %r14
	leaq	(%r9,%r11), %r13
	movslq	-52(%rbp), %r11
	movslq	-84(%rbp), %r9
	vpinsrw	$1, (%rdi,%r15,2), %xmm0, %xmm0
	movq	-584(%rbp), %r15
	imulq	%r11, %rdx
	addq	%r9, %rsi
	vpinsrw	$1, (%rdi,%r15,2), %xmm9, %xmm9
	imulq	%r11, %rcx
	imulq	%r11, %rax
	addq	%r9, %rdx
	vpinsrw	$1, (%rdi,%rdx,2), %xmm8, %xmm8
	movzbl	(%rsi,%rcx), %edx
	movq	-376(%rbp), %rsi
	addq	%r9, %rax
	movq	-600(%rbp), %rcx
	movb	%dl, -337(%rbp)
	vpunpckldq	%xmm9, %xmm0, %xmm0
	vpunpckldq	%xmm8, %xmm3, %xmm3
	vpunpcklqdq	%xmm3, %xmm0, %xmm0
	vmovss	(%rsi,%r14,4), %xmm3
	movq	%rsi, %r14
	vpmovsxwd	%xmm0, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmovaps	%ymm0, -304(%rbp)
	vmovss	(%rsi,%rax,4), %xmm0
	vunpcklps	%xmm0, %xmm3, %xmm0
	vmovss	(%rsi,%rcx,4), %xmm3
	vunpcklps	%xmm7, %xmm3, %xmm3
	vmovss	(%rsi,%r10,4), %xmm7
	movq	-416(%rbp), %rsi
	vmovlhps	%xmm0, %xmm3, %xmm3
	vunpcklps	%xmm6, %xmm7, %xmm6
	vmovss	(%r14,%rsi,4), %xmm7
	movq	-384(%rbp), %rax
	leaq	-344(%rbp), %r14
	movq	-368(%rbp), %rdi
	movq	-456(%rbp), %r8
	movq	%r14, %r15
	movq	-520(%rbp), %rdx
	imulq	%r11, %rax
	addq	%r9, %rax
	movq	-576(%rbp), %r9
	vunpcklps	%xmm2, %xmm7, %xmm0
	vmovlhps	%xmm6, %xmm0, %xmm0
	vmovss	(%rdi,%r13,4), %xmm6
	leaq	-208(%rbp), %r13
	vinsertf128	$0x1, %xmm3, %ymm0, %ymm0
	movq	%r13, %r14
	movq	%rbx, %r13
	vmovaps	%ymm0, -272(%rbp)
	vmovss	(%rdi,%rax,4), %xmm0
	movq	-608(%rbp), %rax
	vmovss	(%rdi,%rax,4), %xmm7
	movswl	-336(%rbp), %eax
	vunpcklps	%xmm0, %xmm6, %xmm0
	vunpcklps	%xmm5, %xmm7, %xmm2
	vmovss	(%rdi,%r9,4), %xmm5
	vmovlhps	%xmm0, %xmm2, %xmm2
	vunpcklps	%xmm4, %xmm5, %xmm4
	vmovss	(%rdi,%r8,4), %xmm5
	vunpcklps	%xmm1, %xmm5, %xmm0
	vmovss	-332(%rbp), %xmm5
	vmovlhps	%xmm4, %xmm0, %xmm0
	vxorps	%xmm4, %xmm4, %xmm4
	vcvtsd2ss	(%rdx), %xmm4, %xmm3
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm0
	vmovaps	%ymm0, -240(%rbp)
	vcvtsi2ssl	%eax, %xmm4, %xmm0
	vmovss	%xmm0, -456(%rbp)
	xorl	%eax, %eax
	movq	%rax, %rbx
	vcvtss2sd	%xmm3, %xmm3, %xmm0
	vaddsd	40(%rdx), %xmm0, %xmm1
	vmovss	%xmm3, -600(%rbp)
	vmovss	%xmm5, -464(%rbp)
	vcvtsd2ss	%xmm1, %xmm1, %xmm4
	vaddsd	48(%rdx), %xmm0, %xmm1
	vaddsd	32(%rdx), %xmm0, %xmm0
	vmovss	%xmm4, -568(%rbp)
	vcvtsd2ss	%xmm1, %xmm1, %xmm4
	vmovss	%xmm4, -576(%rbp)
	vcvtsd2ss	%xmm0, %xmm0, %xmm4
	vmovss	%xmm4, -584(%rbp)
	jmp	.L46
	.p2align 4
	.p2align 3
.L193:
	cmpb	$3, %cl
	je	.L42
	cmpb	$1, %cl
	jne	.L43
	vmovss	-584(%rbp), %xmm6
	vmovss	-272(%rbp,%rbx,4), %xmm5
	vmovss	-240(%rbp,%rbx,4), %xmm4
.L44:
	btq	%rbx, %r13
	vmovss	%xmm4, -384(%rbp)
	vmovss	%xmm5, -376(%rbp)
	vmovss	%xmm6, -368(%rbp)
	vxorps	%xmm0, %xmm0, %xmm0
	jnc	.L45
	vmovss	-304(%rbp,%rbx,4), %xmm2
	leaq	_ZL6ANGLES(%rip), %rcx
	vsubss	-456(%rbp), %xmm2, %xmm1
	vmovss	(%rcx,%rbx,4), %xmm0
	vsubss	-464(%rbp), %xmm0, %xmm0
	vmulss	-540(%rbp), %xmm1, %xmm3
	vmovss	%xmm2, -416(%rbp)
	vmovss	%xmm1, -408(%rbp)
	vmovss	%xmm3, -404(%rbp)
	vzeroupper
	call	cosf@PLT
	vmovss	-416(%rbp), %xmm2
	vmovss	-384(%rbp), %xmm4
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vsubss	-544(%rbp), %xmm2, %xmm2
	vmovss	-404(%rbp), %xmm3
	vmovss	-376(%rbp), %xmm5
	vmovss	-368(%rbp), %xmm6
	vmovss	-408(%rbp), %xmm1
	vmulss	-536(%rbp), %xmm2, %xmm2
	vcvtss2sd	%xmm4, %xmm4, %xmm4
	vcvtss2sd	%xmm6, %xmm6, %xmm6
	vcvtss2sd	%xmm5, %xmm5, %xmm5
	vfmadd132sd	-480(%rbp), %xmm6, %xmm5
	vmovss	.LC6(%rip), %xmm6
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vmulsd	-496(%rbp), %xmm2, %xmm2
	vfmadd132sd	-488(%rbp), %xmm5, %xmm0
	vandps	.LC3(%rip), %xmm3, %xmm5
	vorps	.LC4(%rip), %xmm5, %xmm5
	vfmadd132sd	-472(%rbp), %xmm2, %xmm4
	vmovss	.LC2(%rip), %xmm2
	vfmadd231ss	%xmm3, %xmm3, %xmm2
	vandps	.LC5(%rip), %xmm3, %xmm3
	vsqrtss	%xmm2, %xmm2, %xmm2
	vmulss	-532(%rbp), %xmm2, %xmm2
	vcmpless	%xmm3, %xmm6, %xmm3
	vdivss	%xmm2, %xmm1, %xmm1
	vblendvps	%xmm3, %xmm5, %xmm1, %xmm1
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vfmadd132sd	-504(%rbp), %xmm4, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vxorps	.LC3(%rip), %xmm0, %xmm0
	call	expf@PLT
	vaddss	.LC2(%rip), %xmm0, %xmm0
	vmovss	-548(%rbp), %xmm4
	vdivss	%xmm0, %xmm4, %xmm0
.L45:
	vmovss	%xmm0, (%r14,%rbx,4)
	incq	%rbx
	cmpq	$8, %rbx
	je	.L192
.L46:
	movzbl	(%r15,%rbx), %ecx
	cmpb	$2, %cl
	jne	.L193
	vmovss	-568(%rbp), %xmm6
	vmovss	-272(%rbp,%rbx,4), %xmm5
	vmovss	-240(%rbp,%rbx,4), %xmm4
	jmp	.L44
	.p2align 4
	.p2align 3
.L42:
	vmovss	-576(%rbp), %xmm6
	vmovss	-272(%rbp,%rbx,4), %xmm5
	vmovss	-240(%rbp,%rbx,4), %xmm4
	jmp	.L44
	.p2align 4
	.p2align 3
.L43:
	vxorps	%xmm0, %xmm0, %xmm0
	cmpb	$4, %cl
	je	.L45
	vmovss	-272(%rbp,%rbx,4), %xmm5
	vmovss	-240(%rbp,%rbx,4), %xmm4
	vmovss	-600(%rbp), %xmm6
	jmp	.L44
	.p2align 4
	.p2align 3
.L192:
	vmovdqa	rng(%rip), %ymm1
	vbroadcastss	.LC9(%rip), %ymm2
	movl	$65535, %eax
	vpslld	$13, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vpsrld	$17, %ymm0, %ymm1
	vpxor	%ymm0, %ymm1, %ymm1
	vpslld	$5, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vmovd	%eax, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vmovdqa	%ymm0, rng(%rip)
	vpand	%ymm1, %ymm0, %ymm1
	vpsrld	$16, %ymm0, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vcvtdq2ps	%ymm1, %ymm1
	vfmadd132ps	%ymm2, %ymm1, %ymm0
	vbroadcastss	.LC11(%rip), %ymm1
	vmulps	%ymm1, %ymm0, %ymm0
	vcmpps	$17, -208(%rbp), %ymm0, %ymm0
	vmovmskps	%ymm0, %eax
	testb	%al, %al
	je	.L59
	movzbl	%al, %r15d
	xorl	%r14d, %r14d
	movl	$1, %ecx
	jmp	.L58
	.p2align 4
	.p2align 4
	.p2align 3
.L47:
	incq	%r14
	cmpq	$8, %r14
	je	.L59
.L58:
	btl	%r14d, %r15d
	jnc	.L47
	movq	-448(%rbp), %rax
	movq	-592(%rbp), %rbx
	movslq	(%rax,%r14,4), %rdx
	movslq	32(%rax,%r14,4), %r13
	movq	-528(%rbp), %rax
	cmpq	%rbx, %rax
	je	.L48
	vmovq	%rdx, %xmm6
	addq	$16, %rax
	vpinsrq	$1, %r13, %xmm6, %xmm0
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -528(%rbp)
.L49:
	imulq	%r12, %r13
	movq	-392(%rbp), %rbx
	leaq	63(%rdx,%r13), %rax
	addq	%rdx, %r13
	cmovns	%r13, %rax
	sarq	$6, %rax
	leaq	(%rbx,%rax,8), %rdx
	movq	%r13, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %r13
	andl	$63, %r13d
	subq	%rax, %r13
	jns	.L57
	addq	$64, %r13
	subq	$8, %rdx
.L57:
	shlx	%r13, %rcx, %r13
	incq	-560(%rbp)
	orq	%r13, (%rdx)
	jmp	.L47
.L48:
	movq	-432(%rbp), %rax
	movq	%rbx, %rsi
	movabsq	$576460752303423487, %rdi
	subq	%rax, %rsi
	movq	%rsi, %rax
	sarq	$4, %rax
	cmpq	%rdi, %rax
	je	.L174
	testq	%rax, %rax
	movl	$1, %edi
	cmovne	%rax, %rdi
	addq	%rdi, %rax
	jc	.L52
	movabsq	$576460752303423487, %rdi
	cmpq	%rdi, %rax
	movq	%rdi, %rbx
	cmovbe	%rax, %rbx
	salq	$4, %rbx
.L53:
	movq	%rsi, -376(%rbp)
	movq	%rdx, -368(%rbp)
	movq	%rbx, %rdi
	vzeroupper
	call	_Znwm@PLT
	movq	-368(%rbp), %rdx
	movq	-376(%rbp), %rsi
	movl	$1, %ecx
	movq	-528(%rbp), %r8
	vmovq	%rdx, %xmm3
	vpinsrq	$1, %r13, %xmm3, %xmm0
	vmovdqu	%xmm0, (%rax,%rsi)
	movq	-432(%rbp), %rsi
	cmpq	%rsi, %r8
	je	.L115
	movq	%rax, %rdi
	.p2align 5
	.p2align 4
	.p2align 3
.L55:
	vmovdqu	(%rsi), %xmm0
	addq	$16, %rsi
	addq	$16, %rdi
	vmovdqu	%xmm0, -16(%rdi)
	cmpq	%rsi, %r8
	jne	.L55
	movq	-528(%rbp), %rsi
	movq	-432(%rbp), %rdi
	subq	%rdi, %rsi
	addq	%rax, %rsi
.L54:
	leaq	16(%rsi), %rdi
	movq	%rdi, -528(%rbp)
	movq	-432(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L56
	movq	-592(%rbp), %rsi
	movq	%rax, -376(%rbp)
	movq	%rdx, -368(%rbp)
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	-376(%rbp), %rax
	movl	$1, %ecx
	movq	-368(%rbp), %rdx
.L56:
	addq	%rax, %rbx
	movq	%rax, -432(%rbp)
	movq	%rbx, -592(%rbp)
	jmp	.L49
.L186:
	addq	$64, %rax
	incq	%rsi
	addq	$16, %rcx
	shlx	%rax, %r8, %rax
	orq	%rax, -8(%rdi)
	cmpq	%r14, %rsi
	jb	.L32
	cmpq	$-1, %r14
	jne	.L25
.L187:
	movq	-624(%rbp), %rax
	movl	$8, %r12d
	movq	%rax, -632(%rbp)
	jmp	.L26
	.p2align 4
	.p2align 3
.L62:
	movq	-640(%rbp), %rax
	movq	-624(%rbp), %r14
	movabsq	$1152921504606846975, %rdx
	subq	%rax, %r14
	movq	%r14, %rax
	sarq	$3, %rax
	cmpq	%rdx, %rax
	je	.L175
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L67
	movabsq	$1152921504606846975, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	leaq	0(,%rax,8), %rbx
.L68:
	movq	%rbx, %rdi
	vzeroupper
	call	_Znwm@PLT
.LEHE2:
	movq	%rax, %r13
	movq	-560(%rbp), %rax
	leaq	8(%r14), %r15
	movq	%rax, 0(%r13,%r14)
	testq	%r14, %r14
	jg	.L194
	cmpq	$0, -640(%rbp)
	leaq	0(%r13,%r15), %rax
	movq	%rax, -624(%rbp)
	jne	.L70
.L71:
	leaq	0(%r13,%rbx), %rax
	movq	%r13, -640(%rbp)
	movq	-560(%rbp), %rbx
	movq	%rax, -632(%rbp)
	jmp	.L64
.L188:
	movq	-624(%rbp), %rax
	movq	-632(%rbp), %rbx
	cmpq	%rbx, %rax
	je	.L62
	movq	-440(%rbp), %rbx
	addq	$8, %rax
	movq	%rbx, -8(%rax)
	movq	%rax, -624(%rbp)
.L63:
	movq	-632(%rbp), %r12
	movq	-640(%rbp), %rax
	subq	%rax, %r12
.L26:
	vbroadcasti128	-688(%rbp), %ymm0
	movq	-648(%rbp), %rax
	movq	-664(%rbp), %rbx
	movl	-552(%rbp), %r13d
	movq	$0, 32(%rax)
	movl	$0, 40(%rax)
	movq	$0, 48(%rax)
	movl	$0, 56(%rax)
	movq	$0, 64(%rax)
	vmovdqu	%ymm0, (%rax)
	movq	-392(%rbp), %rax
	subq	%rax, %rbx
	leaq	0(%r13,%rbx,8), %r14
	testq	%r14, %r14
	jne	.L73
	xorl	%ecx, %ecx
	vzeroupper
.L74:
	movl	-552(%rbp), %eax
	movq	-664(%rbp), %rdi
	addq	%rbx, %rcx
	xorl	%edx, %edx
	movl	$1, %r8d
	testl	%eax, %eax
	jne	.L78
	jmp	.L85
	.p2align 6
	.p2align 4
	.p2align 3
.L195:
	incl	%edx
	decq	%r13
	je	.L85
.L78:
	movq	(%rcx), %rsi
	movq	(%rdi), %r10
	shlx	%rdx, %r8, %rax
	andq	%rax, %r10
	movq	%rsi, %r9
	orq	%rax, %r9
	andn	%rsi, %rax, %rax
	testq	%r10, %r10
	cmovne	%r9, %rax
	movq	%rax, (%rcx)
	cmpl	$63, %edx
	jne	.L195
	addq	$8, %rdi
	addq	$8, %rcx
	xorl	%edx, %edx
	decq	%r13
	jne	.L78
.L85:
	movq	-648(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-528(%rbp), %rbx
	vmovdqu	%xmm0, 72(%rax)
	movq	$0, 88(%rax)
	movq	-432(%rbp), %rax
	subq	%rax, %rbx
	je	.L196
	movabsq	$9223372036854775792, %rax
	cmpq	%rbx, %rax
	jb	.L197
	movq	%rbx, %rdi
.LEHB3:
	call	_Znwm@PLT
.LEHE3:
	movq	%rax, %rcx
.L80:
	movq	-648(%rbp), %rax
	addq	%rcx, %rbx
	movq	-432(%rbp), %rsi
	movq	%rbx, 88(%rax)
	movq	-528(%rbp), %rbx
	movq	%rcx, 72(%rax)
	cmpq	%rsi, %rbx
	je	.L89
	subq	%rsi, %rbx
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	call	memcpy@PLT
	movq	%rax, %rcx
	addq	%rbx, %rcx
.L89:
	movq	-648(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-624(%rbp), %rbx
	movq	%rcx, 80(%rax)
	vmovdqu	%xmm0, 96(%rax)
	movq	$0, 112(%rax)
	movq	-640(%rbp), %rax
	subq	%rax, %rbx
	je	.L90
	movabsq	$9223372036854775800, %rax
	cmpq	%rbx, %rax
	jb	.L198
	movq	%rbx, %rdi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rdi
	vmovq	%rax, %xmm3
	movq	-648(%rbp), %rax
	vpunpcklqdq	%xmm3, %xmm3, %xmm0
	leaq	(%rdi,%rbx), %r13
	vmovdqu	%xmm0, 96(%rax)
	movq	%r13, 112(%rax)
	cmpq	$8, %rbx
	jle	.L199
	movq	%rbx, %rdx
	movq	-640(%rbp), %rsi
	call	memcpy@PLT
	cmpq	$0, -392(%rbp)
	movq	-648(%rbp), %rax
	movq	%r13, 104(%rax)
	je	.L99
.L98:
	movq	-656(%rbp), %rsi
	movq	-392(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L99:
	movq	-640(%rbp), %rdi
	movq	%r12, %rsi
	call	_ZdlPvm@PLT
.L100:
	movq	-432(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L9
	movq	-592(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L9:
	movq	-648(%rbp), %rax
	addq	$640, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L194:
	.cfi_restore_state
	movq	-640(%rbp), %rsi
	movq	%r14, %rdx
	movq	%r13, %rdi
	call	memcpy@PLT
	leaq	0(%r13,%r15), %rax
	movq	%rax, -624(%rbp)
.L70:
	movq	-632(%rbp), %rsi
	movq	-640(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	jmp	.L71
.L196:
	xorl	%ecx, %ecx
	jmp	.L80
.L90:
	cmpq	$0, -392(%rbp)
	movq	-648(%rbp), %rax
	movq	$0, 112(%rax)
	je	.L95
	movq	-656(%rbp), %rsi
	movq	-392(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L95:
	cmpq	$0, -640(%rbp)
	je	.L100
	jmp	.L99
	.p2align 4
	.p2align 3
.L115:
	movq	%rax, %rsi
	jmp	.L54
.L73:
	leaq	63(%r14), %rdx
	movq	%rdx, %r15
	movq	%rdx, -360(%rbp)
	shrq	$6, %r15
	salq	$3, %r15
	movq	%r15, %rdi
	vzeroupper
.LEHB5:
	call	_Znwm@PLT
.LEHE5:
	movq	%rax, %rcx
	addq	%rax, %r15
	movq	-648(%rbp), %rax
	testq	%r14, %r14
	movq	%r14, %rdx
	movq	%r15, 64(%rax)
	movq	%rcx, 32(%rax)
	movl	$0, 40(%rax)
	movq	-360(%rbp), %rax
	cmovns	%r14, %rax
	sarq	$63, %rdx
	sarq	$6, %rax
	shrq	$58, %rdx
	leaq	(%rcx,%rax,8), %rsi
	leaq	(%r14,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L200
.L75:
	movq	-648(%rbp), %rdx
	movq	%rsi, 48(%rdx)
	movl	%eax, 56(%rdx)
	cmpq	$8, %rbx
	jle	.L76
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	movq	-392(%rbp), %rsi
	call	memcpy@PLT
	movq	%rax, %rcx
	jmp	.L74
.L185:
	leaq	63(%rax), %rdx
	movq	%rdx, %r13
	movq	%rdx, -360(%rbp)
	shrq	$6, %r13
	salq	$3, %r13
	movq	%r13, %rdi
.LEHB6:
	call	_Znwm@PLT
.LEHE6:
	movq	%rax, %rdi
	movq	%rax, -392(%rbp)
	leaq	(%rax,%r13), %rax
	movq	%rax, -656(%rbp)
	movq	-360(%rbp), %rax
	testq	%rbx, %rbx
	movq	%rbx, %rdx
	cmovns	%rbx, %rax
	sarq	$63, %rdx
	sarq	$6, %rax
	shrq	$58, %rdx
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -664(%rbp)
	leaq	(%rbx,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L22
.L182:
	movq	%r13, %rdx
	xorl	%esi, %esi
	movl	%eax, -552(%rbp)
	movl	$0, -152(%rbp)
	call	memset@PLT
	jmp	.L21
.L22:
	subq	$8, -664(%rbp)
	addl	$64, %eax
	jmp	.L182
.L200:
	addq	$64, %rax
	subq	$8, %rsi
	jmp	.L75
.L110:
	movq	%rax, %rdx
	jmp	.L17
.L197:
.LEHB7:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE7:
.L199:
	movq	-640(%rbp), %rax
	cmpq	$0, -392(%rbp)
	movq	(%rax), %rax
	movq	%rax, (%rdi)
	movq	-648(%rbp), %rax
	movq	%r13, 104(%rax)
	jne	.L98
	jmp	.L99
.L76:
	jne	.L74
	movq	-392(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rcx)
	jmp	.L74
.L15:
	movabsq	$9223372036854775792, %rbx
	jmp	.L16
.L67:
	movabsq	$9223372036854775800, %rbx
	jmp	.L68
	.p2align 4
	.p2align 3
.L198:
.LEHB8:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE8:
.L52:
	movabsq	$9223372036854775792, %rbx
	jmp	.L53
.L171:
	jmp	.L172
.L126:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L106
.L123:
	movq	%rax, %rbx
	jmp	.L102
.L169:
	jmp	.L170
.L122:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L104
.L124:
	movq	%rax, %rbx
	jmp	.L27
.L125:
	movq	%rax, %rbx
	jmp	.L86
	.section	.gcc_except_table,"a",@progbits
.LLSDA10556:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE10556-.LLSDACSB10556
.LLSDACSB10556:
	.uleb128 .LEHB0-.LFB10556
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L169-.LFB10556
	.uleb128 0
	.uleb128 .LEHB1-.LFB10556
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L126-.LFB10556
	.uleb128 0
	.uleb128 .LEHB2-.LFB10556
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L171-.LFB10556
	.uleb128 0
	.uleb128 .LEHB3-.LFB10556
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L122-.LFB10556
	.uleb128 0
	.uleb128 .LEHB4-.LFB10556
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L123-.LFB10556
	.uleb128 0
	.uleb128 .LEHB5-.LFB10556
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L125-.LFB10556
	.uleb128 0
	.uleb128 .LEHB6-.LFB10556
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L124-.LFB10556
	.uleb128 0
	.uleb128 .LEHB7-.LFB10556
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L122-.LFB10556
	.uleb128 0
	.uleb128 .LEHB8-.LFB10556
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L123-.LFB10556
	.uleb128 0
.LLSDACSE10556:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC10556
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold:
.LFSB10556:
.L174:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	leaq	.LC1(%rip), %rdi
	vzeroupper
.LEHB9:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE9:
.L127:
.L172:
	movq	%rax, %rbx
	vzeroupper
.L87:
	movq	-392(%rbp), %rax
	leaq	-160(%rbp), %rdi
	movq	%rax, -160(%rbp)
	movq	-664(%rbp), %rax
	movq	%rax, -144(%rbp)
	movl	-552(%rbp), %eax
	movl	%eax, -136(%rbp)
	movq	-656(%rbp), %rax
	movq	%rax, -128(%rbp)
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movq	-632(%rbp), %rsi
	movq	-640(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L106
.L28:
	movq	-640(%rbp), %rdi
	call	_ZdlPvm@PLT
	jmp	.L106
.L173:
	leaq	.LC1(%rip), %rdi
.LEHB10:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE10:
.L121:
.L170:
	movq	%rax, %rbx
	vzeroupper
.L106:
	movq	-592(%rbp), %rsi
	movq	-432(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L107
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L107:
	movq	%rbx, %rdi
.LEHB11:
	call	_Unwind_Resume@PLT
.LEHE11:
.L175:
	leaq	.LC1(%rip), %rdi
	vzeroupper
.LEHB12:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE12:
.L102:
	movq	-648(%rbp), %rax
	movq	72(%rax), %rdi
	movq	88(%rax), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L180
	vzeroupper
	call	_ZdlPvm@PLT
.L104:
	movq	-648(%rbp), %rdi
	addq	$32, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L87
.L27:
	xorl	%edx, %edx
	xorl	%esi, %esi
	xorl	%ecx, %ecx
	leaq	-160(%rbp), %rdi
	movq	%rsi, -128(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rdx, -144(%rbp)
	movl	%ecx, -136(%rbp)
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movl	$8, %esi
	jmp	.L28
.L86:
	movq	-648(%rbp), %rdi
	addq	$32, %rdi
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L87
.L180:
	vzeroupper
	jmp	.L104
	.cfi_endproc
.LFE10556:
	.section	.gcc_except_table
.LLSDAC10556:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC10556-.LLSDACSBC10556
.LLSDACSBC10556:
	.uleb128 .LEHB9-.LCOLDB13
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L127-.LCOLDB13
	.uleb128 0
	.uleb128 .LEHB10-.LCOLDB13
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L121-.LCOLDB13
	.uleb128 0
	.uleb128 .LEHB11-.LCOLDB13
	.uleb128 .LEHE11-.LEHB11
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB12-.LCOLDB13
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L127-.LCOLDB13
	.uleb128 0
.LLSDACSEC10556:
	.section	.text.unlikely
	.text
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.section	.text.unlikely
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold
.LCOLDE13:
	.text
.LHOTE13:
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I_rng, @function
_GLOBAL__sub_I_rng:
.LFB11646:
	.cfi_startproc
	vmovdqa	.LC14(%rip), %ymm0
	vmovdqa	%ymm0, rng(%rip)
	vzeroupper
	ret
	.cfi_endproc
.LFE11646:
	.size	_GLOBAL__sub_I_rng, .-_GLOBAL__sub_I_rng
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_rng
	.globl	rng
	.bss
	.align 32
	.type	rng, @object
	.size	rng, 32
rng:
	.zero	32
	.section	.rodata
	.align 32
	.type	_ZL5MOVES, @object
	.size	_ZL5MOVES, 64
_ZL5MOVES:
	.long	-1
	.long	-1
	.long	-1
	.long	0
	.long	-1
	.long	1
	.long	0
	.long	-1
	.long	0
	.long	1
	.long	1
	.long	-1
	.long	1
	.long	0
	.long	1
	.long	1
	.align 32
	.type	_ZL6ANGLES, @object
	.size	_ZL6ANGLES, 32
_ZL6ANGLES:
	.long	1075235812
	.long	1078530011
	.long	1081824209
	.long	1070141403
	.long	1083624420
	.long	1061752795
	.long	0
	.long	1085271519
	.set	.LC2,.LC4
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC3:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 16
.LC4:
	.long	1065353216
	.long	0
	.long	0
	.long	0
	.align 16
.LC5:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC6:
	.long	1602224127
	.align 4
.LC9:
	.long	1199570944
	.align 4
.LC11:
	.long	796917760
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC14:
	.long	12345
	.long	67890
	.long	13579
	.long	24680
	.long	11223
	.long	44556
	.long	77889
	.long	99000
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.globl	__gxx_personality_v0
	.ident	"GCC: (Debian 14.2.0-16) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
